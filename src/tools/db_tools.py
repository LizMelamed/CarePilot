from pydantic import BaseModel

from src.db.db_handler import DBHandler
from src.utils.logger import Logger

class GetPatientDataArgs(BaseModel):
    pass


def get_patient_data(db_handler: DBHandler, username: str):
    """
    Get patient's data (birthdate, gender, sex) from db.
    :param db_handler: db handler reference. PASSED BY THE SYSTEM, NOT LLM.
    :param username: the specific user to get patient's data for. PASSED BY THE SYSTEM, NOT LLM.
    :return: the data tuple (birthdate, gender, sex), otherwise None if no user found with the specified username.
    """
    query = """
    SELECT date_of_birth, gender, sex
    FROM users
    WHERE username = ?
    """
    params = (username,)
    result = db_handler.execute(query, params)

    if len(result) == 0:
        return None
    else:
        if len(result) > 1:
            logger = Logger()
            logger.warning(f"Found multiple users with username {username}, num = {len(result)}")

        return result[0]