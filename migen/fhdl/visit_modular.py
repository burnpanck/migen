from copy import copy
from operator import itemgetter
from contextlib import contextmanager
from collections import defaultdict, OrderedDict
from types import MappingProxyType

from migen.fhdl.structure import *
from migen.fhdl.structure import (_Operator, _Slice, _Assign, _ArrayProxy,
                                  _Fragment, _Value, _Statement)




class RegisteringFunc:
    def __init__(self, func, *args, **keywords):
        self.func = func
        self.args = args
        self.keywords = keywords

    def __call__(self, *fargs, **fkeywords):
        newkeywords = self.keywords.copy()
        newkeywords.update(fkeywords)
        return self.func(*(self.args + fargs), **newkeywords)


class RegisteringDict(OrderedDict):
    def __init__(self, mcs, name, bases, *args, **kw):
        super().__init__(*args, **kw)
        self.mcs = mcs
        self.name = name
        self.bases = bases
        self.pre_new_handlers = OrderedDict()
        self.post_new_handlers = OrderedDict()

    def __setitem__(self, key, value):
        if not isinstance(value, RegisteringFunc):
            super().__setitem__(key, value)
            return
        value(self, key)

class CollaborativeMetaclass(type):
    """ Collaborative class creation: objects in the class dict get a chance at customising class creation.

    Whenever an instance of :py:class:`RegisteringFunc` is inserted into the class dict before
    class creation is completed, it will get called with the current class dict as argument and the key
    where it is inserted. Any entry in the attribute `pre_new_handlers` and `post_new_handlers`
    will furthermore be called just before and just after class creation time.
    """
    @classmethod
    def __prepare__(mcs, name, bases):
        ret = RegisteringDict(mcs, name, bases)
        return ret

    def __new__(mcs, name, bases, dct):
        while dct.pre_new_handlers:
            key, handler = dct.pre_new_handlers.popitem(False)
            handler()
        ret = super().__new__(mcs, name, bases, dict(dct))
        while dct.post_new_handlers:
            key, handler = dct.post_new_handlers.popitem(False)
            handler(ret)
        return ret


class WithRegisteredMethods(CollaborativeMetaclass):
    """ Classes of this type may register methods in a type based registry.

    Use :py:meth:`registering_decorator` to generate a decorator that
    registers methods.

    Use :py:meth:`find_handler` to walk a registry to find a maching method.
    """
    def __new__(mcs, name, bases, dct):
        ret = super().__new__(mcs, name, bases, dct)
        opts = {}
        for c in ret.mro():
            o = getattr(c,'default_visitor_options',None)
            if o is None:
                continue
            if isinstance(o,VisitingOptions):
                o = o.all_options
            opts.update((k,v) for k,v in o.items() if v is not None)
        ret.default_visitor_options = VisitingOptions(**opts)
        ret.default_visitor_options.validate()
        return ret


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
            if best_handler is not None:
                return best_handler
        if default is None:
            raise KeyError('Could not an entry in registry "%s" for node of type "%s"'%(registry,type(node).__name__))
        return default

    @staticmethod
    def registering_decorator(registry_name,node_types,**metadata):
        def registerer(dct, key, fun):
            registry = dct.setdefault(registry_name,{})
            for node in node_types:
                registry[node] = fun
            dct[key] = fun

        def decorate(fun):
            for k,v in metadata.items():
                setattr(fun,k,v)
            return RegisteringFunc(registerer,fun=fun)
        return decorate


class VisitingOptions:
    _options = MappingProxyType(dict(
        recurse = frozenset({'before','after',False}),
        recurse_visited = frozenset({False,True}),    # set only when recurse == 'after'
        supply_orig = frozenset({False,True}),
        supply_combined = frozenset({False,True}),
        supply_recursed = frozenset({False,True}),
        return_value = frozenset({'orig','recursed','combine','visited'}),
    ))
    recurse = None
    recurse_visited = None
    supply_orig = None
    supply_combined = None
    supply_recursed = None
    return_value = None

    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

    def __setattr__(self, attr, value):
        opts = type(self)._options.get(attr)
        if opts is None:
            raise AttributeError('Cannot set attribute "%s"'%attr)
        if value not in opts and value is not None:
            raise ValueError('Attribute "%s" must be one of %s, got "%s" instead'%(attr,opts,repr(value)))
        super().__setattr__(attr,value)

    @property
    def all_options(self):
        return {k:getattr(self,k) for k in type(self)._options}

    def validate(self):
        if self.recurse != 'after' and self.recurse_visited is not None:
            raise ValueError('recurse_visited should only be set when recurse="after"')


def update_from_bases(**options):
    def merge_options(dct, key, options):
        # TODO: mirror python's MRO
        for base in dct.bases:
            prev = getattr(base,key,None)
            if prev is None:
                continue
            prev = prev.all_options
        else:
            prev = {}
        prev.update(options)
        options = VisitingOptions(**prev)
        options.validate()
        dct[key] = options
    return RegisteringFunc(merge_options,options=options)

def visitor_for(*node_types,**options):
    """ Decorator marking a method of a NodeTransformer to handle a set of node types.
    """
    registry_name = '_node_visitors'

    def registerer(dct, key, fun):
        registry = dct.setdefault(registry_name, {})
        for node in node_types:
            registry[node] = fun
        dct[key] = fun

    def decorate(fun):
        return RegisteringFunc(registerer, fun=fun)

    return WithRegisteredMethods.registering_decorator(
        '_node_visitors',node_types,
        _options=options,
    )

def recursor_for(*node_types):
    """ Decorator marking a method of a NodeTransformer to be responsible for recursively calling
    `visit` for sub-nodes of nodes of the given set of types.
    """
    return WithRegisteredMethods.registering_decorator('_node_recursors',node_types)

def context_for(*node_types):
    """ Decorator marking a method of a NodeTransformer to be responsible for generating the
    new context values for all child nodes of the given node, and the node itself.
    """
    return WithRegisteredMethods.registering_decorator('_node_context',node_types)

def combiner_for(*node_types):
    """ Decorator marking a method of a NodeTransformer to be responsible for merging the results
    of recursive visitor calls into the input for the visitors.
    """
    return WithRegisteredMethods.registering_decorator('_node_combiners',node_types)

class RecursionResult:
    def __init__(self,node,*args,**kwargs):
        self.node = node
        self.args = args
        self.kwargs = kwargs

class ModularVisitor(metaclass=WithRegisteredMethods):
    """ Abstract modular tree visitor.

    Particularly, the recursion and node-visiting are separated.
    A visitor may choose to let the recursion happen before or after or not at all.
    If pre-recursion is requested, the visitor may optionally receive the
    recursion result as an instance of :py:class:`RecursionResult`.

    The visitation result can be selected too.
    """
    def visit_node(self, orig, node):
        handler = type(self).find_handler('_node_visitors',node,type(self).visit_unknown_node)
        if getattr(handler,'_needs_original_node',None):
            return handler(self,orig,node)
        else:
            return handler(self,node)

    def visit(self, node):
        visitor = type(self).find_handler('_node_visitors',node,type(self).visit_unknown_node)

        opts = visitor._visiting_method

        if not opts.recurse:
            return visitor(self,node)

        orignode = node
        visited = None
        if opts.recurse=='after':
            visited = visitor(self,node)
            if opts.recurse_visited:
                node = visited
        recursed = self.recurse(node)
        if opts.supply_combined or opts.return_value=='combined':
            combined = self.combine(recursed)
        else:
            combined = None
        if opts.recurse=='before':
            args = ()
            if opts.supply_orig:
                args += node,
            if opts.supply_combined:
                args += combined,
            if opts.supply_recursed:
                args += recursed.args
                kw = recursed.kwargs
            else:
                kw = {}
            visited = visitor(self,*args,**kw)
        return dict(
            orig=orignode,
            recursed=recursed,
            combined=combined,
            visited=visited
        )[opts.return_value]

    def recurse(self,node):
        context_gen = type(self).find_handler('_node_context',node)
        recursor = type(self).find_handler('_node_recursors',node)
        with self.subcontext(node, context_gen(self, node)):
            return recursor(self, node)

    def combine(self, recursion_result):
        orig = recursion_result.node
        combiner = type(self).find_handler('_node_combiners',orig)
        return combiner(self, orig, *recursion_result.args, **recursion_result.kwargs)

    @contextmanager
    def subcontext(self,node,context):
        """ Updates contextual variables for visitors of contained nodes.
        Registered context generators may produce additional context values.

        Default implementation generates a list of parents in the context variable `ancestry`,
        and updates any other variable from dictionary supplied by registered context generators.

        .. Warning: This implementation modifies the supplied list in-place
        """
        self.ancestry = ancestry = getattr(self,'ancestry',[])
        ancestry.append(node)
        inexistent = ()   # sentinel
        previous_context = {k:getattr(self,k,inexistent) for k in context}
        try:
            for k,v in context.items():
                setattr(self,k,v)
            yield
        finally:
            ancestry.pop()
            fail = {}
            for k,v in previous_context.items():
                try:
                    if v is inexistent:
                        delattr(self,k)
                    else:
                        setattr(self,k,v)
                except Exception as ex:
                    # Ignore, for now, we still want to revert the rest
                    fail[k] = (v,ex)
            if fail:
                raise RuntimeError('Could not revert context variables: '+str(fail))

    @visitor_for(object)
    def visit_unknown_node(self, node):
        raise TypeError("Don't know how to visit node of type \"%s\""%type(node))

    @context_for(object)
    def context_for_unknown_node(self, node):
        return {}

    @recursor_for(object)
    def recurse_unknown_node(self, node):
        raise TypeError("Don't know how to recurse node of type \"%s\""%type(node))

    @combiner_for(object)
    def combine_unknown_node(self, orig, *args, **kw):
        raise TypeError("Don't know how to combine node of type \"%s\""%type(orig))


class TreeVisitor(ModularVisitor):
    """ Abstract base class for node transformers, similar to :py:class:`.visit.NodeVisitor`, but with more control.

    .. seealso: :py:class:`ModularVisitor`, for instructions how to implement visitors for a particular node
    """
    @combiner_for(object)
    def combine_unknown_node(self, orig, *args, **kw):
        return type(orig)(*args,**kw)

    @recursor_for(Constant, Signal, ClockSignal, ResetSignal)
    def recurse_leaf(self, node):
        return RecursionResult(node)

    @combiner_for(Constant, Signal, ClockSignal, ResetSignal)
    def combine_leaf(self, orig):
        return orig

    @recursor_for(_Operator)
    def recurse_Operator(self, node):
        return RecursionResult(node, node.op, [self.visit(o) for o in node.operands])

    @recursor_for(_Slice)
    def recurse_Slice(self, node):
        return RecursionResult(node,self.visit(node.value), node.start, node.stop)

    @recursor_for(Cat)
    def recurse_Cat(self, node):
        return RecursionResult(node,*[self.visit(e) for e in node.l])

    @recursor_for(Replicate)
    def recurse_Replicate(self, node):
        return RecursionResult(node,self.visit(node.v), node.n)

    @recursor_for(_Assign)
    def recurse_Assign(self, node):
        return RecursionResult(node,self.visit(node.l), self.visit(node.r))

    @recursor_for(If)
    def recurse_If(self, node):
        return RecursionResult(node,
            self.visit(node.cond),
            self.visit(node.t),
            self.visit(node.f),
        )

    @combiner_for(If)
    def combine_If(self, orig, cond, t, f):
        ret = type(orig)(cond)
        ret.t = t
        ret.f = f
        return ret

    @recursor_for(Case)
    def recurse_Case(self, node):
        cases = type(node.sync)(   # preserves order, if cases is an ordered dicts
            (clockname, self.visit(statements))
            for clockname, statements in node.sync.items()
        )
        return RecursionResult(node,self.visit(node.test), cases)

    @recursor_for(_Fragment)
    def recurse_Fragment(self, node):
        sync = type(node.sync)(  # preserves order, if sync is an ordered dicts
            (clockname, self.visit(statements))
            for clockname, statements in node.sync.items()
        )
        return RecursionResult(
            node,
            comb = self.visit(node.comb),
            sync = sync,
        )

    @combiner_for(_Fragment)
    def combine_Fragment(self, orig, comb, sync):
        r = copy(orig)
        r.comb = comb
        r.sync = sync
        return r

    @recursor_for(list, tuple)
    def recurse_sequence(self, node):
        return RecursionResult(node,[self.visit(statement) for statement in node])

    @recursor_for(_ArrayProxy)
    def recurse_ArrayProxy(self, node):
        return RecursionResult(node,
            [self.visit(choice) for choice in node.choices],
            self.visit(node.key)
        )


class AssertConformingTree(TreeVisitor):
    """ Assert that the node structure is conforming.
    """
    ExpressionNodes = (_Value,)
    StatementNodes = (_Statement,)
    StatementSequence = StatementNodes + (_Fragment,list,tuple)

    # TODO: refactor visitors to work with supply_recursed=False,
    # since conceptually, recursion is completely independent of the visitation
    default_visitor_options = dict(
        return_value = 'orig',
        recurse='before',
        supply_orig=True,
        supply_recursed=True,
    )

    def assert_instance(self, node, attr, cls, attrname):
        if not isinstance(attr, cls):
            raise TypeError('Attributes "%s" of nodes of type "%s" must be instances of "%s", got "%s" instead'%(
                attrname, type(node).__name__, str(cls), type(attr).__name__
            ))

    def assert_all_instance(self, node, attr, cls, attrname):
        for k,v in enumerate(attr):
            if not isinstance(attr, cls):
                raise TypeError('All elements of the attributes "%s" of nodes of type "%s" must be instances of "%s", at %d got "%s" instead'%(
                    attrname, type(node).__name__, str(cls), k, type(v).__name__
                ))

    @visitor_for(_Operator)
    def visit_Operator(self, node, op, operands):
        self.assert_all_instance(node, operands, self.ExpressionNodes, "operands")

    @visitor_for(_Slice)
    def visit_Slice(self, node, value, start, stop):
        self.assert_instance(node, value, self.ExpressionNodes, "value")
        self.assert_instance(node, start, int, "start")
        self.assert_instance(node, stop, int, "stop")

    @visitor_for(Cat)
    def visit_Cat(self, node, l):
        self.assert_all_instance(node, l, self.ExpressionNodes, "l")

    @visitor_for(Replicate)
    def visit_Replicate(self, node, v, n):
        self.assert_instance(node, v, self.ExpressionNodes, "v")
        self.assert_instance(node, n, int, "n")

    @visitor_for(_Assign)
    def visit_Assign(self, node, l, r):
        self.assert_instance(node, l, self.ExpressionNodes, "l")
        self.assert_instance(node, r, self.ExpressionNodes, "r")

    @visitor_for(If)
    def visit_If(self, node, cond, t, f):
        self.assert_instance(node, cond, self.ExpressionNodes, "cond")
        self.assert_instance(node, t, self.StatementSequence, "t")
        self.assert_instance(node, f, self.StatementSequence, "f")

    @visitor_for(Case)
    def visit_Case(self, node, test, cases):
        self.assert_instance(node, test, self.ExpressionNodes, "test")
        self.assert_all_instance(node, cases.values(), self.ExpressionNodes, "cases.values()")

    @visitor_for(_Fragment)
    def visit_Fragment(self, node, comb, sync):
        self.assert_instance(node, comb, self.StatementSequence, "comb")
        self.assert_instance(node, sync, dict, "sync")  # maps clock-domains to statement sequences
        self.assert_all_instance(node, sync.values(), self.StatementSequence, "sync.values()")

    @visitor_for(list, tuple, supply_recursed=False)
    def visit_sequence(self, node):
        self.assert_all_instance(node, node, self.StatementSequence, "node")

    @visitor_for(_ArrayProxy)
    def visit_ArrayProxy(self, node, choices, key):
        self.assert_all_instance(node, choices, self.ExpressionNodes, "choices")
        self.assert_instance(node, key, self.ExpressionNodes, "key")


class FormatTree(TreeVisitor):
    default_visitor_options = update_from_bases(
        return_value = 'orig',
        recurse='before',
        supply_recursed=True,
    )

    @visitor_for(_Operator)
    def visit_Operator(self, op, operands):
        if len(operands) == 1:
            return op + operands[0]
        elif len(operands) == 2:
            return '(' + operands[0] + op + operands[1] + ')'
        raise ValueError('cannot handle %d operands on op %s'%(len(operands),op))

    @visitor_for(_Slice)
    def visit_Slice(self, value, start, stop):
        return value+'[%d:%d]'%(start,stop)

class CreateInteractiveTreeView(TreeVisitor):
    """ Creates a static webpage that can be opened with any browser to inspect the tree.

    The tree display is based on `Fanyctree <https://github.com/mar10/fancytree/>`_.
    Or should it be `jsTree <https://www.jstree.com/>`_ instead?
    """
    def __init__(self):
        raise NotImplementedError('That would have been too good to be true anyway')