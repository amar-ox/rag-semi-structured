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


from .data_loader import Loader
from .prompt import Prompt

from .schema_generator import (
    generate_schema_from_sample_chunks,
    extract_steps_schemas,
)
from .script_generator import (
    process_sample_chunks,
    process_sample_chunks_per_section,
)
from .data_extractor import (
    extract_data,
    extract_data_per_section,
)

__all__ = [
    "Prompt",
    "Loader",
    "generate_schema_from_sample_chunks",
    "extract_steps_schemas",
    "process_sample_chunks",
    "process_sample_chunks_per_section",
    "extract_data",
    "extract_data_per_section",
]



