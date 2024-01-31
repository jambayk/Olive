# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union

import onnx
from onnx import AttributeProto, GraphProto, NodeProto
from onnx.shape_inference import infer_shapes_path

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field
from olive.common.utils import onnx_dtype_to_np_dtype

if TYPE_CHECKING:
    from onnx import ModelProto, ValueInfoProto


class SpecialInput(str, Enum):
    """Special inputs for ONNX nodes."""

    INPUT = "__input__"  # user input
    INITIALIZER = "__initializer__"  # constant initializer


class SpecialOutput(str, Enum):
    """Special outputs for ONNX nodes."""

    OUTPUT = "__output__"  # model output


class OnnxNode(ConfigBase):
    """ONNX node."""

    op_type: str
    node: NodeProto  # reference to the node in the model graph
    inputs: List[str]
    outputs: List[str]


class OnnxIO(ConfigBase):
    """ONNX input/output.

    Behaves similar to labeled edges in a graph but can connect to multiple nodes.
    """

    dtype: str = None
    shape: List = None
    source: str = None
    destination: List[str] = Field(default_factory=list)


class OnnxDAG:
    """ONNX graph as a directed acyclic graph (DAG)."""

    def __init__(self):
        self.nodes = {}
        self.ios = {}
        self.graph = defaultdict(list)

    @staticmethod
    def _get_io_type_shape(io: "ValueInfoProto") -> Dict:
        """Get the type and shape of an input/output."""
        tensor_type = io.type.tensor_type
        if tensor_type.elem_type == 0:
            # sequence type
            # TODO(jambayk): add support for different types
            # refer to https://github.com/lutzroeder/netron/blob/main/source/onnx.js#L1424
            tensor_type = io.type.sequence_type.elem_type.tensor_type
        data_type = onnx_dtype_to_np_dtype(tensor_type.elem_type)
        shape = [dim.dim_param if dim.dim_param else dim.dim_value for dim in tensor_type.shape.dim]
        return {
            "dtype": data_type,
            "shape": shape,
        }

    def process_io(self, graph: GraphProto):
        """Process inputs, outputs, initializers, and value_info.

        This will populate the `ios` attribute. Should be called before adding nodes.
        """
        for i in graph.input:
            self.ios[i.name] = OnnxIO(
                source=SpecialInput.INPUT,
                **self._get_io_type_shape(i),
            )
        for o in graph.output:
            self.ios[o.name] = OnnxIO(
                destination=[SpecialOutput.OUTPUT],
                **self._get_io_type_shape(o),
            )
        for initializer in graph.initializer:
            self.ios[initializer.name] = OnnxIO(
                source=SpecialInput.INITIALIZER,
                dtype=onnx_dtype_to_np_dtype(initializer.data_type),
                shape=list(initializer.dims),
            )
        for vi in graph.value_info:
            self.ios[vi.name] = OnnxIO(
                **self._get_io_type_shape(vi),
            )

    def add_node(self, node: NodeProto):
        """Add a node to the graph.

        This adds the node to the `nodes` attribute and connects them using the `ios` attribute.
        """
        name = node.name
        onnx_node = OnnxNode(op_type=node.op_type, node=node, inputs=list(node.input), outputs=list(node.output))
        self.nodes[name] = onnx_node

        for i in node.input:
            self.ios[i].destination.append(name)
            parent = self.ios[i].source
            if parent not in [SpecialInput.INPUT, SpecialInput.INITIALIZER]:
                self.graph[parent].append(name)

        for o in node.output:
            self.ios[o].source = name

    @classmethod
    def from_graph_proto(cls, graph: GraphProto) -> "OnnxDAG":
        """Create an ONNX DAG from a graph proto."""
        dag = cls()
        dag.process_io(graph)
        for node in graph.node:
            dag.add_node(node)
        return dag

    @classmethod
    def from_model_path(cls, model_path: Union[str, Path]) -> Tuple["ModelProto", List["OnnxDAG"]]:
        """Load an ONNX model with shape inference and create a DAG for each graph."""
        with tempfile.NamedTemporaryFile(dir=Path(model_path).parent) as tmpfile:
            shape_infer_model_path = tmpfile.name
            # infer_shapes_path can be used for model >2GB, and infer_shapes cannot.
            infer_shapes_path(model_path, shape_infer_model_path)
            model = onnx.load(shape_infer_model_path)

        dags = []
        graph_queue = [model.graph]
        while graph_queue:
            graph = graph_queue.pop(0)
            dags.append(cls.from_graph_proto(graph))
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == AttributeProto.AttributeType.GRAPH:
                        assert isinstance(attr.g, GraphProto)
                        graph_queue.append(attr.g)
                    if attr.type == AttributeProto.AttributeType.GRAPHS:
                        for g in attr.graphs:
                            assert isinstance(g, GraphProto)
                            graph_queue.append(g)
        return model, dags

    def _topological_sort_util(self, v: str, visited: Set[str], order: List[str]):
        visited.add(v)

        for neighbor in self.graph[v]:
            if neighbor not in visited:
                self._topological_sort_util(neighbor, visited, order)

        order.insert(0, v)

    def topological_sort(self):
        visited = set()
        order = []

        for v in self.nodes:
            if v not in visited:
                self._topological_sort_util(v, visited, order)

        return order
