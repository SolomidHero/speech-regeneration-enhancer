require_relative '../spec_helper'


# ml packages
describe package('torch') do
  it { should be_installed.by('pip').with_version('1.8.0') }
end

describe package('torch-optimizer') do
  it { should be_installed.by('pip').with_version('0.1.0') }
end


# tests
describe package('pytest') do
  it { should be_installed.by('pip').with_version('6.2.2') }
end


# configurations
describe package('omegaconf') do
  it { should be_installed.by('pip').with_version('2.0.6') }
end

describe package('hydra-core') do
  it { should be_installed.by('pip').with_version('1.0.6') }
end


# audio processing and feature extraction
describe package('librosa') do
  it { should be_installed.by('pip').with_version('0.8.0') }
end

describe package('torchaudio') do
  it { should be_installed.by('pip').with_version('0.8.0') }
end

describe package('transformers') do
  it { should be_installed.by('pip').with_version('4.3.2') }
end

describe package('pyworld') do
  it { should be_installed.by('pip').with_version('0.2.12') }
end

describe package('numpy') do
  it { should be_installed.by('pip') }
end

describe package('scipy') do
  it { should be_installed.by('pip') }
end