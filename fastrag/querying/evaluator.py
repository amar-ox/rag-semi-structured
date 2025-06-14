#    FastRAG: Efficient Retrieval Augmented Generation for Semi-structured Data
#    Copyright (C) 2024–2025 Amar Abane
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <https://www.gnu.org/licenses/>.

        
def collect_step1_output_lines(data_dict):
    concatenated_list = []
    for key, data_str in data_dict.items():
        split_data = data_str.split('\n')
        concatenated_list.extend(split_data)
    return concatenated_list
    
def compute_step1_coverage(all_chunks, results, print_missing=False):
    mean_cov = 0
    for file_name, output_data in results.items():
        output_lines = collect_step1_output_lines(output_data)
        stripped_output_lines = [line.strip() for line in output_lines]
        input_lines = "".join(all_chunks[file_name]).split('\n')
        filtered_input_lines = [line.strip() for line in input_lines if line.strip() and line.strip() != '!']

        # Calculate the number of matching lines and track missing lines
        missing_lines = [line for line in filtered_input_lines if line not in stripped_output_lines]

        # Calculate the percentage of matching lines
        matching_lines_count = len(filtered_input_lines) - len(missing_lines)
        if len(filtered_input_lines) > 0:
            matching_percentage = (matching_lines_count / len(filtered_input_lines)) * 100
        else:
            matching_percentage = 0  # To handle case where filtered_input_lines is empty
    
        print(f"--File: {file_name}, Coverage: {matching_percentage:.2f}%")
        mean_cov+=matching_percentage
        if print_missing:
            print("--Missing Lines:")
            for line in missing_lines:
                print(line)
                print("\n")
            print("\n")

    mean_cov = mean_cov / len(results.items())
    print(f"--Mean coverage: {mean_cov:.2f}%")
    return mean_cov


def collect_step2_output_lines(data_dict):
    concatenated_list = []
    for key, dict_list in data_dict.items():
        for item in dict_list:
            input_data = item.get('input_data', '')
            split_data = input_data.split('\n')
            concatenated_list.extend(split_data)
    return concatenated_list


def compute_step2_coverage(file_chunks, results, print_missing=False):
    mean_cov = 0
    for file_name, output_data in results.items():
        output_lines = collect_step2_output_lines(output_data)

        stripped_output_lines = [line.strip() for line in output_lines]

        input_lines = "".join(file_chunks[file_name]).split('\n')

        filtered_input_lines = [line.strip() for line in input_lines if line.strip() and line.strip() != '!']

        # Calculate the number of matching lines and track missing lines
        missing_lines = [line for line in filtered_input_lines if line not in stripped_output_lines]

        # Calculate the percentage of matching lines
        matching_lines_count = len(filtered_input_lines) - len(missing_lines)
        if len(filtered_input_lines) > 0:
            matching_percentage = (matching_lines_count / len(filtered_input_lines)) * 100
        else:
            matching_percentage = 0  # To handle case where filtered_input_lines is empty

        print(f"--File: {file_name}, Coverage: {matching_percentage:.2f}%")
        mean_cov+=matching_percentage
        if print_missing:
            print("--Missing Lines:")
            for line in missing_lines:
                print(line)
                print("\n")
            print("\n")

    mean_cov = mean_cov / len(results.items())
    print(f"--Mean coverage: {mean_cov:.2f}%")
    return mean_cov