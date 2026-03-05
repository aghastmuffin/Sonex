def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)

    if len(b) == 0:
        return len(a)

    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def similarity_percent(a, b):
    distance = levenshtein(a, b)
    max_len = max(len(a), len(b))
    return (1 - distance / max_len) * 100

if __name__ == "__main__":
    str1 = "kitten"
    str2 = "sitting"
    print(f"Levenshtein distance: {levenshtein(str1, str2)}")
    print(f"Similarity: {similarity_percent(str1, str2):.2f}%")