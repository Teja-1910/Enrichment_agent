import re


def clean_webpage_content(page_text: str) -> str:
    """
    Clean rendered webpage text before sending it to the LLM.

    The goal is to remove obvious formatting noise and repeated
    whitespace while preserving useful company information.
    """

    if not page_text:
        return ""

    cleaned_lines = []

    for line in page_text.splitlines():
        cleaned_line = line.strip()

        if not cleaned_line:
            continue

        # Replace repeated whitespace inside a line with one space.
        cleaned_line = re.sub(
            r"[ \t]+",
            " ",
            cleaned_line,
        )

        cleaned_lines.append(cleaned_line)

    cleaned_text = "\n".join(cleaned_lines)

    # Collapse repeated blank lines.
    cleaned_text = re.sub(
        r"\n{2,}",
        "\n",
        cleaned_text,
    )

    return cleaned_text.strip()