# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pynini
from pynini.lib import pynutil, utf8

from data_loader_utils import get_abs_path
from graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from utils import num_to_word

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted. 
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        # integer, negative

        NEMO_CHAR = utf8.VALID_UTF8_CHAR
        NEMO_SIGMA = pynini.closure(NEMO_CHAR)
        NEMO_SPACE = " "
        NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", u"\u00A0").optimize()
        NEMO_NOT_SPACE = pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE).optimize()
        # NEMO_NON_BREAKING_SPACE = u"\u00A0"

        hindi_digit_file = './data/numbers/digit.tsv'
        with open(hindi_digit_file) as f:
            digits = f.readlines()
        hindi_digits = ''.join([line.split()[-1] for line in digits])
        hindi_digits_with_zero = "०" + hindi_digits
        print(f'hindi digits is {hindi_digits}')
        HINDI_DIGIT = pynini.union(*hindi_digits).optimize()
        HINDI_DIGIT_WITH_ZERO = pynini.union(*hindi_digits_with_zero).optimize()

        graph_zero = pynini.string_file("./data/numbers/zero.tsv")
        graph_digit = pynini.string_file("./data/numbers/digit.tsv")
        graph_tens = pynini.string_file("./data/numbers/hindi_tens.tsv")
        # exceptions = pynini.string_file("./data/sentence_boundary_exceptions.txt")

        graph_hundred = pynini.cross("सौ", "")

        graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("०"))
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(graph_tens, pynutil.insert("०") + (graph_digit | pynutil.insert("०")))
        # graph_hundred_component += pynini.union((graph_tens | pynutil.insert("०")) + delete_space + (graph_digit | pynutil.insert("०")),)

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
                pynini.closure(HINDI_DIGIT_WITH_ZERO) + (HINDI_DIGIT_WITH_ZERO - "०") + pynini.closure(HINDI_DIGIT_WITH_ZERO)
        )
        # graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )
        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("हज़ार"),
            pynutil.insert("०००", weight=0.1)
        )

        # fst = graph_thousands
        fst = pynini.union(
            graph_thousands
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )
        fst = fst @ pynini.union(
            pynutil.delete(pynini.closure("०")) + pynini.difference(HINDI_DIGIT_WITH_ZERO, "०") + pynini.closure(
                HINDI_DIGIT_WITH_ZERO), "०"
        )

        # labels_exception = [num_to_word(x) for x in range(0, 5)]
        # graph_exception = pynini.union(*labels_exception)
        #
        # graph = pynini.cdrewrite(pynutil.delete("and"), NEMO_SPACE, NEMO_SPACE, NEMO_SIGMA) @ graph
        #
        self.graph_no_exception = fst
        #
        # #self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph
        self.graph = (pynini.project(fst, "input")) @ fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
