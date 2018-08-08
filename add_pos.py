import re

import requests

from constants import DATA_PATH


def get_pos(sentence):
    response = requests.post("http://text-processing.com/api/tag/", data={"text": sentence, "output": "iob"})
    response_list = response.json().get("text").split("\n")
    return ["_".join(var.split(" ")[0:2]) for var in response_list]


def get_text_with_pos(file_name):
    with open(DATA_PATH + file_name) as f:
        new_file = open(DATA_PATH + "data_with_pos", 'w')
        pattern = re.compile("^\d \d{7}$")
        for i, line in enumerate(f):
            if line == "\n":
                new_file.write("\n")
                continue
            if pattern.match(line):
                continue
            splitted = line.split("\t")
            label = splitted[0]
            words = splitted[1:]
            words_pos = get_pos(words)
            new_file.write(label + "\t")
            for token in words_pos:
                new_file.write(token + " ")
            new_file.write("\n")
