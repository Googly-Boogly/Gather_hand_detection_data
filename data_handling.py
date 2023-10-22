import json
import os


class handle_data:
    def __init__(self, filepath):
        self.filepath = filepath

    def store_data(self, data):
        """
        Append a list of lists to a text file where each line represents one inner list.

        Args:
            filename (str): The name of the text file to save or append the data to.
            data (list): A list of 13 lists, each containing 3 integers.
        """
        with open(self.filepath, 'a') as file:  # Use 'a' for append mode
            # Convert the data to a JSON string and write it to the file
            json_data = json.dumps(data)
            file.write(json_data + '\n')

    def run_txt_to_data(self, line_number):
        """
        takes in the line number and returns the list of lists as lists of lists
        :param line_number: (int) of line number to retrieve
        :return: (list) returns the list of lists of the data for a specifc line
        """
        data = handle_data.txt_file_to_data(self.filepath, line_number)
        if data is not None:
            return data
        else:
            print(f"Line {line_number} not found or has invalid JSON format.")

    def txt_file_to_data(self, line_number):
        """
        Read a specific line from a text file containing JSON data.

        Args:
            filename (str): The name of the text file to read.
            line_number (int): The line number to read (1-based index).

        Returns:
            dict: The parsed JSON data from the specified line.
            None: if invalid json format or line number is out of range

        """
        with open(self.filepath, 'r') as file:
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

    @staticmethod
    def delete_video_data(nums: list):
        """
        take in a list of numbers to delete from the folders data_no_dots_files and data_with_dots_files
        :param nums:
        :return:
        """
        files = os.listdir('total_data/data_with_dots_files')
        for num in nums:
            for filename in files:
                if f"image_data{num}.npy" in filename:
                    file_path = os.path.join('total_data/data_with_dots_files', filename)
                    os.remove(file_path)
        files = os.listdir('total_data/data_no_dots_files')
        for num in nums:
            for filename in files:
                if f"image_data{num}.npy" in filename:
                    file_path = os.path.join('total_data/data_no_dots_files', filename)
                    os.remove(file_path)
        return None

    def count_lines_in_file(self):
        """
        Count the number of lines in a text file.

        Args:
            filename (str): The name of the text file to count lines in.

        Returns:
            int: The number of lines in the file.
        """
        with open(self.filepath, 'r') as file:
            line_count = sum(1 for line in file)
        return line_count

    def delete_last_n_lines(self, n=3):
        """
        Delete the last n (3) lines from a text file.

        Args:
            filename (str): The name of the text file to modify.
            n (int): The number of lines to delete from the end of the file. Default is 3.
        """
        with open(self.filepath, 'r') as file:
            lines = file.readlines()

        keep_lines = lines[:-n]

        with open(self.filepath, 'w') as file:
            file.writelines(keep_lines)

