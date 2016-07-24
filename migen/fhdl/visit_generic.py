from copy import copy
from operator import itemgetter
from contextlib import contextmanager

from migen.fhdl.structure import *
from migen.fhdl.structure import (_Operator, _Slice, _Assign, _ArrayProxy,
                                  _Fragment, _Value, _Statement)


class WithRegisteredMethods(type):
    """ Classes of this type may register methods in a type based registry.

    Use :py:meth:`registering_decorator` to generate a decorator that
    registers methods.

    Use :py:meth:`fund_handler` to walk a registry to find a maching method.
    """
    class RegisteringDict(dict):
        def __setitem__(self, key, value):
            registerer = getattr(value,'_register',None)
            if registerer is not None:
                value = registerer(self,key,value)
            super().__setitem__(key, value)

    @classmethod
    def __prepare__(typ, name, bases):
        ret = typ.RegisteringDict(typ._registries)
        return ret

    def __new__(typ, name, bases, dct):
        return super().__new__(typ,name,bases,dict(dct))

    def find_handler(cls,registry,node,default=None):
        """ Find a handler function in the ancestry of this class for the given node. """
        node_mro = list(reversed(type(node).mro()))
        for trafo_cls in cls.mro():
            specifity = -1
            best_handler = None
            for node_cls, handler in getattr(trafo_cls, registry, {}).items():
                if isinstance(node, node_cls):
                    ns = node_mro.index(node_cls)
                    if ns > specifity:
                        specifity = ns
                        best_handler = handler
            return best_handler
        return default

    @staticmethod
    def registering_decorator(registry_name,node_types):
        def registerer(dct,key,value):
            registry = dct.setdefault(registry_name,{})
            for node in node_types:
                registry[node] = value
            del value._register
            return value
        def decorate(fun):
            fun._register = registerer
            return fun
        return decorate

def visitor_for(*node_types):
    """ Decorator marking a method of a NodeTransformer to handle a set of node types.
    """
    return WithRegisteredMethods.registering_decorator('_node_visitors',node_types)

def recursor_for(*node_types):
    """ Decorator marking a method of a NodeTransformer to be responsible for recursively calling
    `visit` for sub-nodes of nodes of the given set of types.
    """
    return WithRegisteredMethods.registering_decorator('_node_recursors',node_types)


class NodeTransformer(metaclass=WithRegisteredMethods):
    """ Base class for node transformers, similar to :py:class:`.visit.NodeTransformer`, but allows more fine-grained
    control of the recursion process.

    Particularly, the visitors are now only responsible to handle the node at hand, without doing the recursion.
    The recursion is done beforehand.

    If you want to prevent recursion, overload the recursion functions instead.

    Default methods always copy the node, except for:
    - Signals, ClockSignals and ResetSignals
    - Unknown objects
    - All fragment fields except comb and sync
    In those cases, the original node is returned unchanged.
    """
    ExpressionNodes = (_Value,)
    StatementNodes = (_Statement,)
    StructuralNodes = (_Fragment,list,dict)

    def visit_node(self, orig, node, ctxt=None):
        return type(self).find_handler('_node_visitors',node,type(self).visit_unknown_node)(self,orig,node,ctxt)

    def visit(self, node, ctxt=None):
        with self.subcontext(ctxt,node) as subctxt:
            recursed = type(self).find_handler('_node_recursors',node,type(self).recurse_unknown_node)(self,node,subctxt)
        return self.visit_node(node,recursed,ctxt=ctxt)

    def visit_unknown_node(self, orig, node, ctxt):
        return node

    def recurse_unknown_node(self, node, ctxt):
        raise TypeError("Don't know how to recurse into children of nodes of type %s"%type(node))

    @contextmanager
    def subcontext(self,ctxt,node):
        """ Generate context for inner nodes, given an outer context and a current node.

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

    @recursor_for(Constant, Signal, ClockSignal, ResetSignal)
    def recurse_Leaf(self, node, ctxt=None):
        return node

    @recursor_for(_Operator)
    def recurse_Operator(self, node, ctxt=None):
        assert all(isinstance(n,self.ExpressionNodes) for n in node.operands)
        with self.subcontext(ctxt,node) as ctxt:
            return type(node)(node.op, [self.visit(o, ctxt) for o in node.operands])

    @recursor_for(_Slice)
    def recurse_Slice(self, node, ctxt=None):
        assert isinstance(node.value,self.ExpressionNodes)
        assert isinstance(node.start,int)
        assert isinstance(node.stop,int)
        with self.subcontext(ctxt,node) as ctxt:
            return type(node)(self.visit(node.value,ctxt), node.start, node.stop)

    @recursor_for(Cat)
    def recurse_Cat(self, node, ctxt=None):
        assert all(isinstance(n,self.ExpressionNodes) for n in node.l)
        with self.subcontext(ctxt,node) as ctxt:
            return type(node)(*[self.visit(e,ctxt) for e in node.l])

    @recursor_for(Replicate)
    def recurse_Replicate(self, node, ctxt=None):
        assert isinstance(node.v,self.ExpressionNodes)
        assert isinstance(node.n,int)
        with self.subcontext(ctxt,node) as ctxt:
            return type(node)(self.visit(node.v, ctxt), node.n)

    @recursor_for(_Assign)
    def recurse_Assign(self, node, ctxt=None):
        assert isinstance(node.l,self.ExpressionNodes)
        assert isinstance(node.r,self.ExpressionNodes)
        with self.subcontext(ctxt,node) as ctxt:
            return type(node)(self.visit(node.l, ctxt), self.visit(node.r))

    @recursor_for(If)
    def recurse_If(self, node, ctxt=None):
        assert isinstance(node.cond,self.ExpressionNodes)
        assert isinstance(node.t,self.StatementNodes)
        assert isinstance(node.f,self.StatementNodes)
        with self.subcontext(ctxt,node) as ctxt:
            r = type(node)(self.visit(node.cond, ctxt))
            r.t = self.visit(node.t, ctxt)
            r.f = self.visit(node.f, ctxt)
        return r

    @recursor_for(Case)
    def recurse_Case(self, node, ctxt=None):
        assert isinstance(node.test,self.ExpressionNodes)
        assert all(isinstance(n,self.ExpressionNodes) for n in node.cases.values())
        with self.subcontext(ctxt,node) as ctxt:
            cases = {v: self.visit(statements, ctxt)
                     for v, statements in sorted(node.cases.items(),
                                                 key=lambda x: str(x[0]))}
            r = type(node)(self.visit(node.test, ctxt), cases)
            return r

    @recursor_for(_Fragment)
    def recurse_Fragment(self, node, ctxt=None):
        assert isinstance(node.comb,self.StructuralNodes+self.StatementNodes)
        assert isinstance(node.sync,self.StructuralNodes+self.StatementNodes)
        r = copy(node)
        with self.subcontext(ctxt,node) as ctxt:
            r.comb = self.visit(node.comb, ctxt)
            r.sync = self.visit(node.sync, ctxt)
        return r

    # NOTE: this will always return a list, even if node is a tuple
    @recursor_for(list, tuple)
    def recurse_statements(self, node, ctxt=None):
        assert all(isinstance(n,self.StatementNodes) for n in node)
        with self.subcontext(ctxt,node) as ctxt:
            return [self.visit(statement, ctxt) for statement in node]

    @recursor_for(dict)
    def recurse_clock_domains(self, node, ctxt=None):
        assert all(isinstance(n,self.StatementNodes) for n in node.values())
        with self.subcontext(ctxt,node) as ctxt:
            return {
                clockname: self.visit(statements, ctxt)
                for clockname, statements in sorted(
                    node.items(),
                    key=itemgetter(0)
                )
            }

    @recursor_for(_ArrayProxy)
    def recurse_ArrayProxy(self, node, ctxt=None):
        assert isinstance(node.key,self.ExpressionNodes)
        assert all(isinstance(n,self.ExpressionNodes) for n in node.choices)
        with self.subcontext(ctxt,node) as ctxt:
            return type(node)(
                [self.visit(choice, ctxt) for choice in node.choices],
                self.visit(node.key, ctxt)
            )
