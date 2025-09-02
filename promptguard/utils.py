import importlib

from ._log import get_logger

LOGGER = get_logger()


def lazy_load_dep(import_name: str, package_name: str | None = None):
    """Helper function to lazily load optional dependencies. If the dependency is not
    present, the function will raise an error _when used_.

    NOTE: This wrapper adds a warning message at import time.
    """

    if package_name is None:
        package_name = import_name

    spec = importlib.util.find_spec(import_name)  # type: ignore
    if spec is None:
        LOGGER.warning(
            f"Optional feature dependent on missing package: {import_name} was initialized.\n"
            f"Use `pip install {package_name}` to install the package if running locally."
        )

    return importlib.import_module(import_name)


def get_nlp_model(language: str):
    """Helper function to load SpaCy language models."""

    LANGUAGES = {
        "en": "en_core_web_lg",
        "de": "de_core_news_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
        "it": "it_core_news_lg",
        "pt": "pt_core_news_lg",
    }

    return LANGUAGES[language] if language in LANGUAGES.keys() else language
