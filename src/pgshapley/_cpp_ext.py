try:
    from pgshapley._core import PreparedTreeData, explain_trees

    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False
    explain_trees = None
    PreparedTreeData = None
