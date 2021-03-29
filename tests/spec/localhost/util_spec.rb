require_relative '../spec_helper'


describe file('/usr/local/lib/pkgconfig/sndfile.pc') do
  it { should exist }
end


describe file('/proc/driver/nvidia/version') do
  it { should exist }
end


describe package('nvcc') do
  it { should be_installed }
end
