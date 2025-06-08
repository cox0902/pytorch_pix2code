
from typing import *

from functools import partial

import torch


def default_label_formatter(n: "TreeNode",
                            vocabs: List[str],
                            show_id: bool = False):
    label = vocabs[n.iv]
    if show_id:
        label += f"#{n.id[-3:]}"
    return label 


class TreeNode:

    def __init__(self, 
                 iv: Union[int, torch.Tensor], 
                 parent: Optional["TreeNode"] = None,
                 device: Optional[torch.device] = None):
        self.id = hex(id(self))
        self.set_iv(iv, device=device)

        # print(self.iv, device, self.iv.device)

        self.parent = parent

        self.children: List[TreeNode] = []
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.height = 1 if parent is None else parent.height + 1
        self.score = 0
        self.count = 1

    def __repr__(self):
        return f"{self.iv}#{self.id[-3:]}"

    def set_iv(self, 
               iv: Union[int, torch.Tensor], 
               device: Optional[torch.device] = None):
        if device is None:
            self.iv = iv
        else:
            if type(iv) == int:
                self.iv = torch.tensor(iv).to(device)
            else:
                self.iv = iv.clone().to(device)

    def get_iv_premitive(self) -> int:
        if type(self.iv) == int:
            return self.iv
        return self.iv.item()

    def add_child(self, 
                  iv: Union[int, torch.Tensor], 
                  device: Optional[torch.device] = None) -> "TreeNode":
        node = TreeNode(iv, parent=self, device=device)
        self.children.append(node)
        return node
    
    def delete(self):
        for i, each in enumerate(self.parent.children):
            if each.id == self.id:
                break
        self.parent.children = [
            *self.parent.children[:i], 
            *self.children,
            *self.parent.children[i + 1:]]
        
        for each in self.children:
            each.parent = self.parent

    def insert(self, 
               iv, 
               device: Optional[torch.device] = None):
        node = TreeNode(iv, parent=self, device=device)
        node.children = self.children
        self.children = [node]
        for each in node.children:
            each.parent = node
        return node
            
    #

    def preorder_walk(self, visit: Callable[["TreeNode"], Any]):
        visit(self)
        for each in self.children:
            each.preorder_walk(visit)

    def preorder_walk_children(self, visit: Callable[["TreeNode"], Any]):
        for each in self.children:
            each.preorder_walk(visit)

    def postorder_walk(self, visit: Callable[["TreeNode"], Any]):
        for each in self.children:
            each.postorder_walk(visit)
        visit(self)

    def postorder_walk_children(self, visit: Callable[["TreeNode"], Any]):
        for each in self.children:
            each.postorder_walk(visit)

    #

    @staticmethod
    def _build_list(n: "TreeNode", ivs: List[int]):
        ivs.append(n.get_iv_premitive())
        if len(n.children) > 0:
            ivs.append(5)  # [LB]
            for each in n.children:
                if each.get_iv_premitive() in [3, 4]:
                    continue
                TreeNode._build_list(each, ivs)
            ivs.append(6)  # [RB]
            
    def build_list(self) -> List[int]:
        ivs = [3]  # [START]
        for each in self.children:
            if each.get_iv_premitive() in [3, 4]:
                continue
            TreeNode._build_list(each, ivs)
        ivs.append(4)  # [END]
        return ivs
    
    def equal(self, that: "TreeNode") -> bool:
        return self.build_list() == that.build_list()
    
    #
    
    @staticmethod
    def _make_graph(n: "TreeNode", 
                    dot,
                    formatter,
                    show_connections: bool = False,
                    show_controls: bool = True):
        
        if show_controls or (not show_controls and n.iv >= 8):
            dot.node(name=n.id, label=formatter(n))

        with dot.subgraph() as s:
            s.attr(rank="same")
            children = n.children if show_controls else n.children[1:-1]
            for each in children:
                s.node(name=each.id, label=formatter(each))
                if show_controls or (not show_controls and n.iv >= 8):
                    dot.edge(n.id, each.id)

        if show_connections:
            children = n.children[:-1] if show_controls else n.children[1:-2]
            for each in children:
                dot.edge(each.id, each.right.id, style="dashed", arrowhead="none", arrowtail="none")

        for each in n.children:
            TreeNode._make_graph(each, 
                                 dot,
                                 formatter, 
                                 show_connections=show_connections, 
                                 show_controls=show_controls)

    def visualize(self, 
                  vocabs: List[str], 
                  show_id: bool = False,
                  show_connections: bool = False,
                  show_controls: bool = True):
        from graphviz import Digraph
        dot = Digraph()
        label_formatter = partial(default_label_formatter, 
                                  vocabs=vocabs,
                                  show_id=show_id)
        TreeNode._make_graph(self, 
                             dot, 
                             label_formatter,
                             show_connections=show_connections,
                             show_controls=show_controls)
        return dot

    #

    def clone(self, device: Optional[torch.device] = None) -> "TreeNode":
        return TreeNode.build_tree(self.build_list(), device)
   
    @staticmethod
    def build_tree(code, 
                   device: Optional[torch.device] = None,
                   verbose: bool = False) -> "TreeNode":
        
        root = TreeNode(3, device)
        node = root
        queue: List[TreeNode] = [node]
        for ii, iv in enumerate(code):
            if iv == 3:  # [START]
                continue
            if iv == 4:  # [END]
                break
            if iv == 5:  # [LB]
                queue.append(node)
                continue
            if iv == 6:  # [RB]
                if len(queue) == 0:
                    if verbose:
                        print("warning: unclosed [LB]")
                    break
                queue.pop()
                continue
            assert iv != 0
            if len(queue) == 0:
                if verbose:
                    print("warning: unclosed [RB]")
                break
            node = queue[-1].add_child(iv)
        return root