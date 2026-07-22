from src.db.supabase_handler import SupabaseHandler
from src.tools.db_tools import GetPatientDataArgs
from src.tools.tool import ToolRepository
from src.utils.logger import Logger


class System:

    def __init__(self):
        self._logger: Logger = Logger()
        self._logger.info("Initializing System...")
        self._db_handler = SupabaseHandler()
        self._tools = ToolRepository()

        self._logger.info("Registering tools...")
        self._register_tools()

        self._logger.info("System Initialized.")

    def _register_tools(self):

        # EXPECTS username IN HIDDEN CONTEXT.
        def baked_get_patient_data(username: str):
            return self._db_handler.get_patient_data(username)

        self._tools.register_func(
            baked_get_patient_data,
            name="get_patient_data",
            description="Get patient's data (date of birth, gender, sex) from the database.",
            args_schema=GetPatientDataArgs
        )