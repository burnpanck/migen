""" vhdl.types -- Represent VHDL types in python
"""

import abc

class VHDLType(abc.ABC):
    _identity = ('name',)

    def __eq__(self,other):
        if not isinstance(other,VHDLType):
            return NotImplemented
        return type(other) == type(self) and self._prehash == other._prehash

    @property
    def _prehash(self):
        return tuple(getattr(self,attr) for attr in type(self)._identity)

    def __hash__(self):
        return hash(self._prehash)

    @abc.abstractmethod
    def equivalent_to(self,other):
        pass

    @abc.abstractmethod
    def castable_to(self,other):
        pass

    def compatible_with(self,other):
        """ Can be assigned without a cast. """
        return self.ultimate_base == other.ultimate_base

    unconstrained = False

    @property
    def ultimate_base(self):
        return self

    @abc.abstractproperty
    def vhdl_repr(self):
        pass

class VHDLSubtype(abc.ABC):
    """ Mixin for subtypes
    """
    def __init__(self,base,name=None,*a,**kw):
        self.base = base
        super(VHDLSubtype,self).__init__(name,*a,**kw)

    @property
    def ultimate_base(self):
        return self.base.ultimate_base

class VHDLAccess(VHDLType):
    pass
class VHDLFile(VHDLType):
    pass

class VHDLScalar(VHDLType):
    @abc.abstractproperty
    def length(self):
        pass

class VHDLInteger(VHDLScalar):
    _identity = ('name','left','right','ascending')
    def __init__(self,name,left=None,right=None,ascending=None):
        assert (left is None) == (right is None)
        if ascending is None:
            if left is not None:
                ascending = not left>right
        else:
            assert left is not None
            assert left == right or ascending == (left<right)
        self.name = name
        self.left = left
        self.right = right
        self.ascending = ascending

    @property
    def vhdl_repr(self):
        return self.name

    def __str__(self):
        return '<%s:%s range %d %s %d>'%(self.name,self.ultimate_base.name,self.left,'to' if self.ascending else 'downto',self.right)

    @property
    def low(self):
        return min(self.left,self.right)

    @property
    def high(self):
        return max(self.left,self.right)

    @property
    def length(self):
        return max(0,
            (self.right - self.left + 1)
            if self.ascending else
            (self.left - self.right + 1)
        )

    def constrain(self,left,right,ascending=None,name=None):
        if self.left is not None:
            assert self.low <= min(left,right) and max(left,right)<=self.high
        return VHDLSubInteger(self,name,left,right,ascending)

    def equivalent_to(self,other):
        return isinstance(other,VHDLInteger) and self.left==other.left and self.right==other.right and self.ascending==other.ascending

    def castable_to(self,other):
        # TODO: check if these are the correct requirements (does ascending need to match?)
        return isinstance(other,VHDLInteger) and self.left>=other.left and self.right<=other.right and self.ascending==other.ascending

class VHDLSubInteger(VHDLSubtype,VHDLInteger):
    @property
    def vhdl_repr(self):
        if self.name is not None:
            return self.name
        return (
            self.base.name +
            ' range ' +
            str(self.left) +
            (' to ' if self.ascending else ' downto ') +
            str(self.right)
        )

class VHDLReal(VHDLScalar):
    pass

class VHDLEnumerated(VHDLScalar):
    _identity = ('name','values')
    def __init__(self,name,values=()):
        self.name = name
        self.values = tuple(values)

    @property
    def vhdl_repr(self):
        return self.name

    def __str__(self):
        return '<%s:%s(%s)>'%(self.name,self.ultimate_base.name,', '.join(str(v) for v in self.values))

    @property
    def left(self):
        return self.values[0]

    @property
    def right(self):
        return self.values[-1]

    @property
    def length(self):
        return len(self.values)

    def equivalent_to(self,other):
        return isinstance(other,VHDLEnumerated) and self.values==other.values

    def castable_to(self,other):
        return self.compatible_with(other)

class VHDLSubEnum(VHDLSubtype,VHDLEnumerated):

    def vhdl_repr(self):
        raise NotImplementedError

class VHDLComposite(VHDLType):
    pass

class VHDLArray(VHDLComposite):
    _identity = ('name','valuetype','indextypes')
    def __init__(self,name,valuetype,*indextypes):
        self.name = name
        self.valuetype = valuetype
        self.indextypes = indextypes

    @property
    def vhdl_repr(self):
        return self.name

    def __str__(self):
        return '<%s:%s array (%s) of %s>'%(self.name,self.ultimate_base.name,', '.join(str(v) for v in self.indextypes),self.valuetype)

    def constrain(self,name=None,*indextypes):
        return VHDLSubArray(self,name,*indextypes)

    def __getitem__(self,item):
        if isinstance(item,slice):
            return self[(item,)]
        if not len(self.indextypes) == len(item):
            raise IndexError('You must specify a constraint for every index in an array specification')
        return self.constrain(None,*[
            t.constrain(
                s.start if s.start is not None else t.left,
                s.stop if s.stop is not None else t.right,
            )
            for t,s in zip(self.indextypes,item)
        ])

    def equivalent_to(self,other):
        return (
            isinstance(other,VHDLArray)
            and self.valuetype.equivalent_to(other.valuetype)
            and len(self.indextypes) == len(other.indextypes)
            and all(s.equivalent_to(o) for s,o in zip(self.indextypes,other.indextypes))
        )

    def castable_to(self, other):
#        print(self,' castable_to ',other,'?')
        if not isinstance(other, VHDLArray) or not self.valuetype.castable_to(other.valuetype):
#            print(self.valuetype,' not castable_to ',isinstance(other,VHDLArray) and other.valuetype,'!')
            return False
        if len(self.indextypes) != len(other.indextypes):
#            print(self.indextypes,' not the same dimensionality as ',other.indextypes,'!')
            return False
#        print('|'.join(str(s.castable_to(o)) for s,o in zip(self.indextypes, other.indextypes)))
        return all(s.castable_to(o) and s.length == o.length for s,o in zip(self.indextypes, other.indextypes))

    def compatible_with(self,other):
        if not isinstance(other,VHDLArray) or not self.ultimate_base == other.ultimate_base:
            return False
        return self.unconstrained or other.unconstrained or tuple(i.length for i in self.indextypes) == tuple(i.length for i in other.indextypes)

    unconstrained = True

class VHDLSubArray(VHDLSubtype,VHDLArray):
    unconstrained = False

    def __init__(self,base,name,*indextypes):
        if not len(base.indextypes) == len(indextypes) or not all(
            s.compatible_with(b) for s,b in zip(
                indextypes, base.indextypes
            )
        ):
            raise TypeError('The index of an array subtype must be compatible with the index of the base array')
        self.base = base
        self.name = name
        self.indextypes = indextypes

    @property
    def vhdl_repr(self):
        if self.name is not None:
            return self.name
        return (
            self.base.name +
            ' ( ' + ', '.join(
                idx.vhdl_repr for idx in self.indextypes
            ) + ' )'
        )


    def constrain(self,name=None,*indextypes):
        raise TypeError('Cannot constrain already constrained arrays')

    @property
    def valuetype(self):
        return self.base.valuetype

class VHDLRecord(VHDLComposite):
    pass


# - Standard types
bit = VHDLEnumerated('bit',(0,1))
boolean = VHDLEnumerated('boolean',(False,True))
character = VHDLEnumerated('character',tuple(chr(k) for k in range(32,128))) # TODO: verify correct set

integer = VHDLInteger('integer',-2**31+1,2**31-1)
natural = VHDLInteger('natural',0,2**31-1)
positive = VHDLInteger('positive',1,2**31-1)

severity_level = VHDLEnumerated('severity_level','note warning error failure'.split())

bit_vector = VHDLArray('bit_vector',bit,natural)
string = VHDLArray('string',character,natural)



# - std_logic_1164
std_ulogic = VHDLEnumerated('std_ulogic',tuple('UX01ZWLH-'))
std_logic = VHDLEnumerated('std_logic',tuple('UX01ZWLH-')) # TODO: implement resolution behaviour?

std_ulogic_vector = VHDLArray('std_ulogic_vector',std_ulogic,natural)
std_logic_vector = VHDLArray('std_logic_vector',std_logic,natural)
signed = VHDLArray('signed',std_logic,natural)
unsigned = VHDLArray('unsigned',std_logic,natural)
