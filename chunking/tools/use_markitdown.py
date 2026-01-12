import subprocess

def convert_html_to_markdown(input_file, output_file):
    """
    Convert an HTML file to Markdown using the markitdown tool.

    Args:
    input_file (str): The path to the input HTML file.
    output_file (str): The path to the output Markdown file.

    Returns:
    bool: True if the conversion was successful, False otherwise.
    """
    try:
        result = subprocess.run(
            ['markitdown', input_file, '-o', output_file],
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print("Command output:", result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: The markitdown command was not found. Please ensure that the tool is correctly installed and added to your system's PATH.")
        return False
    except Exception as e:
        print(f"An unknown error occurred.: {str(e)}")
        return False
