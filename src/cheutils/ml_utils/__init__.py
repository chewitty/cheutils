from cheutils.properties_util import AppProperties
from cheutils.debugger import Debugger

APP_PROPS = AppProperties()
prop_key = 'project.models.supported'
MODELS_SUPPORTED = APP_PROPS.get_list(prop_key)
assert (MODELS_SUPPORTED is not None), 'Models supported must be specified'
DEBUGGER = Debugger()
DEBUGGER.debug('Models supported = ', MODELS_SUPPORTED)