import csv


def get_chosen_speaker(path):
    chosen_nationalities = ["USA","UK","Canada","Australia"]
    nationalities = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            new_row = row[0].split()
            if i > 0:
                nationalities.append((new_row[1],new_row[3]))
            i += 1

    speakerlist = []
    for j in range(len(nationalities)):
        if nationalities[j][1] in chosen_nationalities:
            speakerlist.append(nationalities[j][0])

    return speakerlist


if __name__ == '__main__':
    file_path = "EmoVoxCeleb_meta.csv"
    speakerlist = get_chosen_speaker(file_path)
    print(speakerlist[:20])
    print(len(speakerlist))
