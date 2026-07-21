from src.db.libsql_handler import LibSqlHandler
from src.tools.db_tools import get_patient_data, GetPatientDataArgs
from src.tools.tool import ToolRepository
from src.utils.logger import Logger


class System:

    def __init__(self):
        self._logger: Logger = Logger()
        self._logger.info("Initializing System...")
        self._db_handler = LibSqlHandler()
        self._tools = ToolRepository()

        self._logger.info("Registering tools...")
        self._register_tools()

        self._logger.info("System Initialized.")

    def _register_tools(self):

        # EXPECTS username IN HIDDEN CONTEXT.
        def baked_get_patient_data(username: str):
            return get_patient_data(db_handler=self._db_handler, username=username)

        self._tools.register_func(
            baked_get_patient_data,
            name="get_patient_data",
            description="Get patient's data (date of birth, gender, sex) from the database.",
            args_schema=GetPatientDataArgs
        )