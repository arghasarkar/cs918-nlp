import json

file_path = "signal-news1/signal-news1.jsonl"

with open(file_path, "r") as f:
    i = 0
    for line in f:
        print(line)
        parsed_json_line = json.loads(line)
        print("Printing content")
        print(parsed_json_line["content"])

        with open("output.txt", "w") as w:
            w.write("----line--\n")
            w.write(line)
            w.write("----parsed---\n")
            w.write(parsed_json_line["content"])
            w.write("\n--done--")
        if i >= 0:
            break
        i += 1

neg.add_word("fair")
neg.add_word("fame")
neg.add_word("famous")
neg.add_word("fair")
neg.add_word("fair")
neg.add_word("famous")
neg.add_word("absurd")
neg.add_word("abuse")