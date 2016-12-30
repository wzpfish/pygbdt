#coding: utf-8

class Dataset(object):
  """Class to restore the data and provide some convenient
  functions to access the targets and features.
  """
  def __init__(self, filename):
    self.num_features = 0
    self.samples = self._load_data(filename)
    self.sample_indices = list(range(0, len(self.samples[0])))
    self.feature_indices = list(range(0, self.num_features))

  def _load_data(self, filename):
    samples = [[], []] #labels, features
    with open(filename) as fin:
      for line in fin:
        line = line.rstrip('\n')
        if not line: continue
        fields = line.split('\t')
        label = float(fields[0])
        features = [float(it) for it in fields[1:]]
        if self.num_features == 0:
          self.num_features = len(features)
        elif self.num_features != len(features):
          raise ValueError('Every sample must have same dimension of features. Expected dim %s, got dim %s'
            % (self.num_features, len(features)))
        samples[0].append(label)
        samples[1].append(features)
    return samples

  def get_targets(self):
    return self.samples[0]

  def get_target_at(self, index):
    return self.samples[0][index]
    
  def get_sample_indices(self):
    return self.sample_indices

  def get_features(self):
    return self.samples[1]

  def get_feature_indices(self):
    return self.feature_indices

  def get_feature_values_of_samples(self, sample_indices, feature_idx):
    feature_values = []
    for sample_idx in sample_indices:
      value = self.samples[1][sample_idx][feature_idx]
      feature_values.append(value)
    return feature_values


# Test codes.
def gen_dummy_file():
  content = ('1\t0.1\t0.2\t0.3\t0.4\n'
         '0\t0.5\t0.6\t0.7\t0.8\n'
         '0\t0.9\t1.0\t1.1\t1.2\n')
  dummy_filename = './dummy.txt'
  with open(dummy_filename, 'w') as fout:
    fout.write(content)
  return dummy_filename


def test_test():
  dummy_filename = gen_dummy_file()
  dataset = Dataset(dummy_filename)
  import os
  os.remove(dummy_filename)
  
  sample_indices = dataset.get_sample_indices()
  assert sample_indices == [0, 1, 2]
  feature_indices = dataset.get_feature_indices()
  assert feature_indices == [0, 1, 2, 3]
  feature_values = dataset.get_feature_values_of_samples([0, 1, 2], 0)
  assert feature_values == [0.1, 0.5, 0.9]
  feature_values = dataset.get_feature_values_of_samples([0, 1, 2], 3)
  assert feature_values == [0.4, 0.8, 1.2]
  print('Test passed.')


if __name__ == '__main__':
  test_test()
