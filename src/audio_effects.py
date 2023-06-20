def flanger(samples):
    samples_copy = []
    for index, sample in enumerate(samples):
        samples_copy.append(sample)
        if index % 500 == 0:
            samples_copy.append(sample)

    return samples + samples_copy[:len(samples)]
