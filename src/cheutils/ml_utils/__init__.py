from cheutils.properties_util import PropertiesUtil
from cheutils.debugger import Debugger
DEBUGGER = Debugger()
APP_PROPS = PropertiesUtil()
prop_key = 'project.namespace'
PROJECT_NAMESPACE = APP_PROPS.get(prop_key)
assert (PROJECT_NAMESPACE is not None), 'Project namespace property must be specified'
prop_key = 'project.model.supported'
MODELS_SUPPORTED = APP_PROPS.get_list(prop_key)
assert (MODELS_SUPPORTED is not None), 'Models supported must be specified'
DEBUGGER.debug('Models supported = ', MODELS_SUPPORTED)