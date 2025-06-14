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


from setuptools import setup, find_packages
#import nltk

# Custom NLTK setup hook
#def download_nltk_resources():
#    resources = ['punkt', 'stopwords', 'punkt_tab']
#    for resource in resources:
#        nltk.download(resource)

#download_nltk_resources()


setup(
    name="fastrag",
    version="0.1",
    packages=find_packages(),
)