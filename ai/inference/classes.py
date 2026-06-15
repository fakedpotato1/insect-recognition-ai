CLASS_NAMES = (
    "ant",
    "bed-bug",
    "bee",
    "beetle",
    "bernsteinschabe",
    "cockroach",
    "fly",
    "fruitfly",
    "grasshopper",
    "hornet",
    "housefly",
    "ladybug",
    "mosquito",
    "moth",
    "silverfish",
    "slug",
    "snail",
    "spider",
    "tiger mosquito",
    "wasp",
)


def display_name(label):
    return str(label).replace("-", " ").title()


def class_metadata(raw_label, encoded_index):
    label_text = str(raw_label)

    try:
        class_id = int(label_text)
    except ValueError:
        class_id = None

    if class_id is not None and 0 <= class_id < len(CLASS_NAMES):
        label = CLASS_NAMES[class_id]
        return class_id, label, display_name(label)

    if label_text in CLASS_NAMES:
        class_id = CLASS_NAMES.index(label_text)
        return class_id, label_text, display_name(label_text)

    return int(encoded_index), label_text, display_name(label_text)
