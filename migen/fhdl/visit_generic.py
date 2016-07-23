from copy import copy
from operator import itemgetter
from contextlib import contextmanager

from migen.fhdl.structure import *
from migen.fhdl.structure import (_Operator, _Slice, _Assign, _ArrayProxy,
                                  _Fragment, _Value, _Statement)


class WithRegisteredVisitors(type):
    @classmethod
    def __prepare__(typ, name, bases):
        class ActiveDict(dict):
            def __setitem__(typ,key,value):
                for cls in getattr(value,'_visits_nodes',()):
                    typ['_visitors'][cls] = value
                super().__setitem__(key,value)
        ret = ActiveDict()
        ret['_visitors'] = dict()
        return ret
    def __new__(typ, name, bases, dct):
        return super().__new__(typ,name,bases,dict(dct))

def visitor_for(*node_types):
    """ Decorator marking a method of a NodeTransformer to handle a set of node types.
    """
    def decorate(fun):
        fun._visits_nodes = node_types
        return fun
    return decorate


class NodeTransformer(metaclass=WithRegisteredVisitors):
    """ Base class for node transformers, similar to :py:class:`.visit.NodeTransformer`, but with context.

    Default methods always copy the node, except for:
    - Signals, ClockSignals and ResetSignals
    - Unknown objects
    - All fragment fields except comb and sync
    In those cases, the original node is returned unchanged.
    """
    ExpressionNodes = (_Value,)
    StatementNodes = (_Statement,)
    StructuralNodes = (_Fragment,list,dict)

    def visit(self, node, ctxt=None):
        node_mro = list(reversed(type(node).mro()))
        for trafo_cls in type(self).mro():
            specifity = -1
            best_visitor = None
            for node_cls,visitor in trafo_cls._visitors.items():
                if isinstance(node,node_cls):
                    ns = node_mro.index(node_cls)
                    if ns > specifity:
                        specifity = ns
                        best_visitor = visitor
            if best_visitor is not None:
                return best_visitor(self, node, ctxt)
        self.visit_unknown(node, ctxt)

    @contextmanager
    def subcontext(self,ctxt,node):
        """ Generic visitor to generate context for inner nodes, given an outer context and a current node.

        Default implementation generates a list of parents

        .. Warning: This implementation modifies the supplied list in-place
        """
        if ctxt is None:
            ctxt = []
        ctxt.append(node)
        try:
            yield ctxt
        finally:
            ctxt.pop()

    @visitor_for(Constant, Signal, ClockSignal, ResetSignal)
    def visit_Leaf(self, node, ctxt=None):
        return node

    @visitor_for(_Operator)
    def visit_Operator(self, node, ctxt=None):
        assert all(isinstance(n,self.ExpressionNodes) for n in node.operands)
        with self.subcontext(ctxt,node) as ctxt:
            return _Operator(node.op, [self.visit(o, ctxt) for o in node.operands])

    @visitor_for(_Slice)
    def visit_Slice(self, node, ctxt=None):
        assert isinstance(node.value,self.ExpressionNodes)
        assert isinstance(node.start,int)
        assert isinstance(node.stop,int)
        with self.subcontext(ctxt,node) as ctxt:
            return _Slice(self.visit(node.value,ctxt), node.start, node.stop)

    @visitor_for(Cat)
    def visit_Cat(self, node, ctxt=None):
        assert all(isinstance(n,self.ExpressionNodes) for n in node.l)
        with self.subcontext(ctxt,node) as ctxt:
            return Cat(*[self.visit(e,ctxt) for e in node.l])

    @visitor_for(Replicate)
    def visit_Replicate(self, node, ctxt=None):
        assert isinstance(node.v,self.ExpressionNodes)
        assert isinstance(node.n,int)
        with self.subcontext(ctxt,node) as ctxt:
            return Replicate(self.visit(node.v, ctxt), node.n)

    @visitor_for(_Assign)
    def visit_Assign(self, node, ctxt=None):
        assert isinstance(node.l,self.ExpressionNodes)
        assert isinstance(node.r,self.ExpressionNodes)
        with self.subcontext(ctxt,node) as ctxt:
            return _Assign(self.visit(node.l, ctxt), self.visit(node.r))

    @visitor_for(If)
    def visit_If(self, node, ctxt=None):
        assert isinstance(node.cond,self.ExpressionNodes)
        assert isinstance(node.t,self.StatementNodes)
        assert isinstance(node.f,self.StatementNodes)
        with self.subcontext(ctxt,node) as ctxt:
            r = If(self.visit(node.cond, ctxt))
            r.t = self.visit(node.t, ctxt)
            r.f = self.visit(node.f, ctxt)
        return r

    @visitor_for(Case)
    def visit_Case(self, node, ctxt=None):
        assert isinstance(node.test,self.ExpressionNodes)
        assert all(isinstance(n,self.ExpressionNodes) for n in node.cases.values())
        with self.subcontext(ctxt,node) as ctxt:
            cases = {v: self.visit(statements, ctxt)
                     for v, statements in sorted(node.cases.items(),
                                                 key=lambda x: str(x[0]))}
            r = Case(self.visit(node.test, ctxt), cases)
            return r

    @visitor_for(_Fragment)
    def visit_Fragment(self, node, ctxt=None):
        assert isinstance(node.comb,self.StructuralNodes+self.StatementNodes)
        assert isinstance(node.sync,self.StructuralNodes+self.StatementNodes)
        r = copy(node)
        with self.subcontext(ctxt,node) as ctxt:
            r.comb = self.visit(node.comb, ctxt)
            r.sync = self.visit(node.sync, ctxt)
        return r

    # NOTE: this will always return a list, even if node is a tuple
    @visitor_for(list, tuple)
    def visit_statements(self, node, ctxt=None):
        assert all(isinstance(n,self.StatementNodes) for n in node)
        with self.subcontext(ctxt,node) as ctxt:
            return [self.visit(statement, ctxt) for statement in node]

    @visitor_for(dict)
    def visit_clock_domains(self, node, ctxt=None):
        assert all(isinstance(n,self.StatementNodes) for n in node.values())
        with self.subcontext(ctxt,node) as ctxt:
            return {
                clockname: self.visit(statements, ctxt)
                for clockname, statements in sorted(
                    node.items(),
                    key=itemgetter(0)
                )
            }

    @visitor_for(_ArrayProxy)
    def visit_ArrayProxy(self, node, ctxt=None):
        assert isinstance(node.key,self.ExpressionNodes)
        assert all(isinstance(n,self.ExpressionNodes) for n in node.choices)
        with self.subcontext(ctxt,node) as ctxt:
            return _ArrayProxy(
                [self.visit(choice, ctxt) for choice in node.choices],
                self.visit(node.key, ctxt)
            )

    def visit_unknown(self, node, ctxt):
        return node