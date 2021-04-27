import pynini
from pynini.lib import pynutil, utf8

from graph_utils import delete_space

if __name__=='__main__':

    NEMO_CHAR = utf8.VALID_UTF8_CHAR
    NEMO_SIGMA = pynini.closure(NEMO_CHAR)

    hindi_digit_file = './data/numbers/digit.tsv'
    with open(hindi_digit_file) as f:
        digits = f.readlines()
    hindi_digits = "०" + ''.join([line.split()[-1] for line in digits])
    print(f'hindi digits is {hindi_digits}')
    HINDI_DIGIT = pynini.union(*hindi_digits).optimize()

    graph_zero = pynini.string_file("./data/numbers/zero.tsv")
    graph_digit = pynini.string_file("./data/numbers/digit.tsv")
    graph_tens = pynini.string_file("./data/numbers/hindi_tens.tsv")

    #print("graph digit fst is ", graph_digit)

    graph_hundred = pynini.cross("सौ", "")

    graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("०"))
    graph_hundred_component += delete_space

    # graph_hundred_component += pynini.union(
    #     graph_teen | pynutil.insert("00"),
    #     (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
    # )

    graph_hundred_component += pynini.union(graph_tens, pynutil.insert("०") + delete_space + (graph_digit | pynutil.insert("०")))

    # graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
    #     pynini.closure(HINDI_DIGIT) + (HINDI_DIGIT - "०") + pynini.closure(HINDI_DIGIT)
    # )

    graph_thousands = pynini.union(
        graph_hundred_component + delete_space + pynutil.delete("हज़ार"),
        pynutil.insert("०००", weight=0.1),
    )

    #fst = graph_thousands
    fst = pynini.union(
        graph_thousands
        + graph_hundred_component,
        graph_zero,
    )
    fst = fst @ pynini.union(
        pynutil.delete(pynini.closure("०")) + pynini.difference(HINDI_DIGIT, "०") + pynini.closure(HINDI_DIGIT), "०"
    )

    fst = fst.optimize()
    fst = pynini.cdrewrite(fst, "", "", NEMO_SIGMA)
    fst = fst.optimize()

    file_path = 'sample_input.txt'

    with open(file_path) as f:
        lines = f.readlines()

    print("Printing output lines \n")
    for line in lines:

        s = pynini.escape(line.strip())
        ans = s @ fst
        #print("***********")
        #print("ans is \n")
        #print(ans)
        astr = pynini.shortestpath(ans).string()
        print(f'Original: {s} Output: {astr}')