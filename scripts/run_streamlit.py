import os
import subprocess

from src.common.logging import logger, setup_logging

setup_logging()


def launch_web_interface():
    """
    Invokes the Streamlit process for the local demo application.

    This script simplifies the process of starting the UI by automatically
    locating and running the Streamlit app file.
    """
    logger.info("Launching Streamlit demo environment")

    app_logic_path = os.path.join("src", "streamlit_app", "app.py")

    if not os.path.exists(app_logic_path):
        logger.error("Streamlit application script missing", path=app_logic_path)
        return

    try:
        subprocess.run(["streamlit", "run", app_logic_path], check=True)
    except Exception as runtime_error:
        logger.error("Streamlit runtime failure", error=str(runtime_error))


if __name__ == "__main__":
    launch_web_interface()
