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


# fastrag/data_extractor.py

def extract_data(file_chunks, extractor):
    results = {}
    for file_name, chunks in file_chunks.items():
        data = "\n".join(chunks)
        results[file_name] = extractor(data)
    return results


def extract_data_per_section(step1_results, section_extractors):
    results = {}
    for file, section_data in step1_results.items():
        print(f"Processing file {file}")
        file_results = {}
        for section, data in section_data.items():
            # time.sleep(1)
            if data and section not in section_extractors:
                print(f"Skip section {section}")
            elif data:  # call extractor of section
                print(f"Processing section {section}")
                # print(section_extractors[section](data))
                file_results[section] = section_extractors[section](data)
            else:
                print(f"Skip section {section}: EMPTY")

        results[file] = file_results
    return results
