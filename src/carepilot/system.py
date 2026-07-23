from src.db.supabase_handler import SupabaseHandler
from src.tools.db_tools import GetPatientDataArgs, GetFileArgs, ListFilesArgs, QueryFileArgs
from src.tools.tool import ToolRepository
from src.utils.logger import Logger

# top_k chunks retrieved by query_file()
TOP_K = 5

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

        self._tools.register_func(
            afunc=self._db_handler.get_patient_data,
            name="get_patient_data",
            description="Get patient's data (date of birth, gender, sex) from the database.",
            args_schema=GetPatientDataArgs
        )

        self._tools.register_func(
            afunc=self._db_handler.get_file,
            name="get_file",
            description="Retrieves the contents of the specified file of the user from database.",
            args_schema=GetFileArgs
        )

        self._tools.register_func(
            afunc=self._db_handler.list_files,
            name="list_files",
            description="Retrieves a list of file names owned by the user from database.",
            args_schema=ListFilesArgs
        )

        async def baked_query_file(username: str, query: str):
            return await self._db_handler.query_file(username, query, top_k=TOP_K)

        self._tools.register_func(
            afunc=baked_query_file,
            name="query_file",
            description=f"Retrieves a list of top_k={TOP_K} file chunks (file_name, chunk_index, chunk_text) "
                        "from files owned by the user, from database. This is based on a query written in natural language."
                        "each chunk is based on the content of the file.",
            args_schema=QueryFileArgs
        )