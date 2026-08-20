import os
import re
from langchain.tools import Tool
from langchain_ollama import ChatOllama

def _analysis_tool(detections_file: str) -> str:
    """
    Reads the YOLO detection log (detections_file) and uses the DeepSeek model via ChatOllama
    to produce an email-style summary in plain text with:

    - Number of 'Danger' detections
    - Brief explanation
    - General comment/warning

    The final text is written to 'analysis.txt', no chain-of-thought or placeholders.
    """
    if not os.path.exists(detections_file):
        return f"detections_file not found: {detections_file}"

    # Read YOLO detection log
    with open(detections_file, "r") as f:
        detections_text = f.read().strip()

    # Count Danger occurrences
    manual_count = detections_text.count("Danger")

    # Build system and human prompts
    system_prompt = (
        "You are an AI that summarizes YOLO detection logs. "
        "Return only the final summary with 3 bullet lines:\n"
        "1) - Number of 'Danger' detections: X\n"
        "2) - Brief explanation: ...\n"
        "3) - General comment/warning: ...\n"
        "No chain of thought, no subject, no email headers. Output ONLY the body lines in plain text."
    )

    human_prompt = f"""
YOLO detection log:
{detections_text}

Please produce exactly 3 bullet lines:
- Number of 'Danger' detections: <number>
- Brief explanation: <one line summary>
- General comment/warning: <safety note>
"""

    llm = ChatOllama(
        model="deepseek-r1:1.5b",
        temperature=0.0,
        base_url="http://localhost:11434",
    )

    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]

    # Get raw model output
    ai_msg = llm.invoke(messages)
    raw_output = ai_msg.content.strip()

    # Remove <think> blocks
    raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)

    # Split lines
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]

    # We'll look for the bullet lines
    danger_idx = None
    explain_idx = None
    comment_idx = None

    for i, line in enumerate(lines):
        lower_line = line.lower()
        if "number of 'danger' detections:" in lower_line:
            danger_idx = i
        elif "brief explanation:" in lower_line:
            explain_idx = i
        elif "general comment/warning:" in lower_line:
            comment_idx = i

    final_lines = lines[:]

    # 1) If Danger line doesn't exist, append fallback
    if danger_idx is None:
        final_lines.append(f"- Number of 'Danger' detections: {manual_count}")
    else:
        # If it exists but has placeholder <number>, replace it
        line_text = final_lines[danger_idx]
        if "<number>" in line_text:
            new_line = line_text.replace("<number>", str(manual_count))
            final_lines[danger_idx] = new_line

    # 2) Explanation line fallback
    if explain_idx is None:
        if manual_count == 0:
            final_lines.append("- Brief explanation: No Danger found.")
        else:
            final_lines.append("- Brief explanation: Potential risks detected.")
    else:
        # If it has <one line summary> placeholder
        if "<one line summary>" in final_lines[explain_idx]:
            final_lines[explain_idx] = final_lines[explain_idx].replace(
                "<one line summary>",
                "Potential risks detected." if manual_count else "No Danger found."
            )

    # 3) Comment line fallback
    if comment_idx is None:
        final_lines.append("- General comment/warning: Review and investigate for safety issues.")
    else:
        if "<safety note>" in final_lines[comment_idx]:
            final_lines[comment_idx] = final_lines[comment_idx].replace(
                "<safety note>", "Please monitor closely."  # or custom
            )

    # Join final lines
    final_output = "\n".join(final_lines)

    # Save to analysis.txt
    analysis_path = os.path.join(os.path.dirname(detections_file), "analysis.txt")
    with open(analysis_path, "w") as out_f:
        out_f.write(final_output)

    return f"Analysis complete. Summary saved to '{analysis_path}'."

analysis_tool = Tool(name="analysis_tool", func=_analysis_tool, return_direct=True)
