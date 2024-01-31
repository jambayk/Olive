# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.onnx.triton_fusion import Fusion, OnnxDAG
from olive.passes.pass_config import PassConfigParam


class AutoFusion(Pass):
    """Automatically fuse nodes in an ONNX model using auto-generated custom operators."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "min_occurrence": PassConfigParam(
                type_=int,
                default_value=10,
                description="Minumum number of occurance of a fusion pattern to be considered for fusion.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        onnx_model, dags = OnnxDAG.from_model_path(model.model_path)

        # get fusable chains
        fusable_chains = defaultdict(list)
        for dag_idx, dag in enumerate(dags):
            chains = self.get_fusable_chains(dag)
            for node_names, node_types in chains.values():
                dtype = self.check_shapes_and_types(dag, node_names)
                if dtype is not None:
                    fusable_chains[(dtype, tuple(node_types))].append((dag_idx, node_names))

        # only consider chains that occur more than min_occurrence times
        for node_types in list(fusable_chains.keys()):
            if len(fusable_chains[node_types]) < config["min_occurrence"]:
                del fusable_chains[node_types]

        # order chains by occurrence and length
        # Matmul chains are given higher priority
        ordered_chain_types = sorted(
            fusable_chains.keys(),
            key=lambda x: (x[1][0] == "MatMul", len(fusable_chains[x]), len(x[1])),
            reverse=True,
        )

        # fuse chains
        fusions = []
        for dtype, chain_type in ordered_chain_types:
            fusion = Fusion(dtype, chain_type[0], list(chain_type[1:]))
            num_fused = 0
            for dag_idx, node_names in fusable_chains[(dtype, chain_type)]:
                node_protos = dags[dag_idx].get_node_protos(node_names)
                if not node_protos:
                    continue
                fused_node = fusion.fuse_nodes(node_protos)
                dags[dag_idx].replace_nodes(node_names, fused_node)
                num_fused += 1
            fusions.append((fusion, num_fused))

    @classmethod
    def _get_fusable_chains_util(
        cls, dag: OnnxDAG, v: str, visited: Set[str], chains: Dict[str, Tuple[List[str], List[str]]]
    ) -> None:
        """Find fusable chains from all nodes reachable from v."""
        if v in visited:
            return

        visited.add(v)
        for neighbor in dag.connections[v]:
            if neighbor not in visited:
                cls._get_fusable_chains_util(dag, neighbor, visited, chains)

        node = dag.nodes[v]
        # check if node can be a base op
        # we only consider nodes with a single output
        if not Fusion.is_valid_base_op(node.op_type) or len(dag.connections[v]) != 1:
            return

        child = dag.connections[v][0]
        child_node = dag.nodes[child]
        if not Fusion.is_valid_fused_op(child_node.op_type):
            return

        if child in chains:
            chains[v] = ([v, *chains[child][0]], [node.op_type, *chains[child][1]])
        else:
            chains[v] = ([v, child], [node.op_type, child_node.op_type])

    @classmethod
    def get_fusable_chains(cls, dag: OnnxDAG) -> Dict[str, Tuple[List[str], List[str]]]:
        """Return fusable chains in the graph.

        There will be overlap between chains. For example, A -> B -> C and B -> C will both be returned.
        A -> B -> C and D -> C is also possible. The priority of the chains during fusion will be determined
        by the fusion rules and heuristics.

        :param dag: The ONNX graph.
        :return: A dictionary of fusable chains. Key is the base op and value is a tuple (op_names, op_types).
        """
        chains = {}
        visited = set()
        for v in dag.nodes:
            cls._get_fusable_chains_util(dag, v, visited, chains)
        return chains

    @staticmethod
    def is_broadcastable(a_shape: List[Union[str, int]], b_shape: List[Union[str, int]]) -> bool:
        """Check if two shapes are broadcastable.

        Broadcasting support is currently limited to the following unidirectional constraints:
            - shape of second input must be a suffix of the shape of the first input
            - Only leading 1s are allowed in the shape of the second input
            - Example [2, 3, 4, 5]: [1], [5], [1, 5], [4, 5], ...

        :param a_shape: The shape of the first input.
        :param b_shape: The shape of the second input.
        :return: True if the shapes are broadcastable, False otherwise.
        """
        if len(b_shape) > len(a_shape):
            return False

        leading_ones = True
        mismatched_dims = False
        for a, b in zip(a_shape[-len(b_shape) :], b_shape):
            if leading_ones and b == 1:
                continue
            leading_ones = False
            if a != b:
                mismatched_dims = True
                break

        return not mismatched_dims

    @classmethod
    def check_shapes_and_types(cls, dag: OnnxDAG, node_names: List[str]) -> Optional[str]:
        """Check if the chain is valid for fusion.

        Rules:
            - Date type of the inputs and outputs must be the same
            - Single input nodes are always valid
            - Non-commutative ops must have the previous output as the first input
            - The other input must be broadcastable to the output of the base op
        Assumes each node has at most two inputs and one output.

        :param dag: The ONNX graph.
        :param node_names: The names of the nodes in the chain.
        :return: Data type of the chain if valid, None otherwise.
        """
        # base node is the first node in the chain
        base = node_names[0]
        dtype = dag.get_input_dtypes(base)[0]
        a_shape = dag.get_output_shapes(base)[0]

        for node_idx, name in enumerate(node_names):
            # check if the data type is the same
            if not all(dtype == dt for dt in dag.get_input_dtypes(name) + dag.get_output_dtypes(name)):
                return None

            # check if the shapes are broadcastable
            # skip base node
            if node_idx == 0:
                continue

            inputs = dag.nodes[name].inputs
            if len(inputs) == 1:
                continue
            if len(inputs) > 2:
                # should not reach since we only consider two input nodes
                return None

            prev_output = dag.nodes[node_names[node_idx - 1]].outputs[0]
            if not Fusion.is_commutative_op(dag.nodes[name].op_type) and dag.nodes[name].inputs[0] != prev_output:
                # output is not the first input
                return None

            connection_idx = dag.nodes[name].inputs.index(prev_output)
            b_shape = dag.ios[inputs[1 - connection_idx]].shape
            if not cls.is_broadcastable(a_shape, b_shape):
                return None

        return dtype
