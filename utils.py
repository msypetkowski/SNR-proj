def group(iterable, count):
    i = iter(iterable)
    next_list = []
    while True:
        try:
            if len(next_list) < count:
                next_list.append(next(i))
            else:
                if len(next_list) > 0:
                    yield next_list
                next_list = []
        except StopIteration:
            break
    if next_list:
        yield next_list


def transposed_group(iterable, count):
    return (zip(*gr) for gr in group(iterable, count))
