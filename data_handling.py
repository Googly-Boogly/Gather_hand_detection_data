import json


def run_txt_to_data(line_number):
    """
    takes in the line number and returns the list of lists as lists of lists
    :param line_number: (int) of line number to retrieve
    :return: (list) returns the list of lists of the data for a specifc line
    """
    filename = 'data.txt'
    data = txt_file_to_data(filename, line_number)

    if data is not None:
        # print(data)
        # print(len(data))
        return data
    else:
        print(f"Line {line_number} not found or has invalid JSON format.")


def txt_file_to_data(filename, line_number):
    """
    Read a specific line from a text file containing JSON data.

    Args:
        filename (str): The name of the text file to read.
        line_number (int): The line number to read (1-based index).

    Returns:
        dict: The parsed JSON data from the specified line.
        None: if invalid json format or line number is out of range

    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        if 1 <= line_number <= len(lines):
            line_content = lines[line_number - 1].strip()
            try:
                data = json.loads(line_content)
                return data
            except json.JSONDecodeError:
                return None  # Invalid JSON format
        else:
            return None  # Line number is out of range


def count_lines_in_file(filename='data.txt'):
    """
    Count the number of lines in a text file.

    Args:
        filename (str): The name of the text file to count lines in.

    Returns:
        int: The number of lines in the file.
    """
    with open(filename, 'r') as file:
        line_count = sum(1 for line in file)
    return line_count


def delete_last_n_lines(filename='data.txt', n=3):
    """
    Delete the last n (3) lines from a text file.

    Args:
        filename (str): The name of the text file to modify.
        n (int): The number of lines to delete from the end of the file. Default is 3.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    keep_lines = lines[:-n]

    with open(filename, 'w') as file:
        file.writelines(keep_lines)

