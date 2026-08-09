from src.tools.db_tools import (
    GetFileArgs,
    GetPatientDataArgs,
    GetPatientDocumentsArgs,
    ListFilesArgs,
    QueryClinicalRagArgs,
    QueryFileArgs,
    query_clinical_rag_tool,
)
from src.tools.text_tools import (
    DraftMessageArgs,
    SummarizationArgs,
    message_drafting,
    summarization,
)
from src.tools.tool import ToolRepository
from src.utils.logger import Logger

# top_k chunks retrieved by query_file()
TOP_K = 5

class System:

    def __init__(self, db_handler=None):
        self._logger: Logger = Logger()
        self._logger.info("Initializing System...")
        if db_handler is None:
            from src.db.db_handler import get_db_handler

            db_handler = get_db_handler()
        self._db_handler = db_handler
        self._tools = ToolRepository()

        self._logger.info("Registering tools...")
        self._register_tools()

        self._logger.info("System Initialized.")

    def _register_tools(self):

        # username isn't specified by the LLM.

        self._tools.register_func(
            afunc=self._db_handler.get_patient_data,
            name="get_patient_data",
            description=(
                "Get the patient's basic demographic profile (date of birth, gender, sex) from the database. "
                "USE FOR: questions about the patient's own basic profile info, e.g. 'How old am I?' or "
                "'What's my date of birth on file?'. DO NOT USE FOR: medical records, appointments, labs, "
                "referrals, medications, or insurance -- those live in documents, not this profile."
            ),
            args_schema=GetPatientDataArgs
        )

        self._tools.register_func(
            afunc=self._db_handler.get_file,
            name="get_file",
            description=(
                "Retrieves the full contents of ONE specific file, given its exact file name. "
                "USE FOR: the patient (or a prior step) already named an exact file, e.g. "
                "'show me labs_2026_03_10.txt' or 'open my medication_schedule.txt file'. "
                "DO NOT USE FOR: searching or browsing across documents when you don't already know the exact "
                "file name -- use query_file or get_patient_documents for that instead."
            ),
            args_schema=GetFileArgs
        )

        self._tools.register_func(
            afunc=self._db_handler.list_files,
            name="list_files",
            description=(
                "Lists the file NAMES owned by the patient -- no content, no categories. "
                "USE FOR: 'What documents/files do I have?' or 'How many files are on my account?' when the "
                "patient only wants an inventory, not the content of any document. "
                "DO NOT USE FOR: answering a question about what's IN a document -- use get_patient_documents "
                "or query_file for that."
            ),
            args_schema=ListFilesArgs
        )

        self._tools.register_func(
            afunc=self._db_handler.get_patient_documents,
            name="get_patient_documents",
            description=(
                "Retrieves patient-owned documents (full content, unfiltered) directly from the database. "
                "USE FOR: browsing several recent documents at once, or when the patient names or implies a "
                "document without giving its exact file name, e.g. 'what's the status of my referral?' or "
                "'show me my recent lab results'. "
                "DO NOT USE FOR: open-ended or content-specific questions (e.g. 'did the doctor say anything "
                "about my dosage?') -- prefer query_file for those, since it searches inside document content."
            ),
            args_schema=GetPatientDocumentsArgs
        )

        async def baked_query_file(username: str, query: str):
            return await self._db_handler.query_file(username, query, top_k=TOP_K)

        self._tools.register_func(
            afunc=baked_query_file,
            name="query_file",
            description=(
                f"Semantic search over the patient's own files: returns the top_k={TOP_K} most relevant chunks "
                "(file_name, chunk_index, chunk_text) for a natural-language query, searching inside document "
                "content. "
                "USE FOR: open-ended or content-specific questions where the right document isn't obvious, "
                "e.g. 'what did my last labs say about my hemoglobin?' or 'did my doctor mention anything "
                "about a dosage change?'. "
                "DO NOT USE FOR: an exact known file name (prefer get_file)."
            ),
            args_schema=QueryFileArgs
        )

        self._tools.register_func(
            afunc=query_clinical_rag_tool,
            name="query_clinical_rag",
            description="Retrieves source-attributed clinical reference chunks for a general clinical question.",
            args_schema=QueryClinicalRagArgs
        )

        self._tools.register_func(
            afunc=summarization,
            name="summarization",
            description="Summarizes source text in plain language without adding diagnosis or medical advice.",
            args_schema=SummarizationArgs
        )

        self._tools.register_func(
            afunc=message_drafting,
            name="message_drafting",
            description="Drafts in-scope patient messages for care logistics, insurance, referrals, and records.",
            args_schema=DraftMessageArgs
        )

    def get_tools(self):
        return self._tools.get_all_tools()

    def get_db_handler(self):
        return self._db_handler
