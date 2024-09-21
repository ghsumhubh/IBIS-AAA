# IBIS AAA
 Our implementation of the AAA models for the IBIS challenge


Take test file (IBIS.test_data.Final.v1)
Extract the contents (as files) to 
/data/Test/
Specifically for CHS GHTS HTS and only .fasta


Take test file(IBIS.train_data.Final.v1)
Extract the HTS folder to 
/data/



Run preprocess.py
Run HTS.py

HTS -> X predictions should be at 
output/X/HTS/output_v35_1_2.tsv.gz



VERSIONS


Python 3.9.18

absl-py                      2.0.0
appdirs                      1.4.4
astunparse                   1.6.3
Babel                        2.9.1
biopython                    1.82
blivet                       3.6.0
Brlapi                       0.8.2
cachetools                   5.3.2
certifi                      2023.7.22
cffi                         1.14.5
chardet                      4.0.0
charset-normalizer           3.3.2
click                        8.1.7
cloudpickle                  3.0.0
cockpit                      311.2
conda                        4.14.0
conda-package-handling       1.7.3
contourpy                    1.2.0
cryptography                 36.0.1
cupshelpers                  1.0
cutadapt                     4.6
cycler                       0.12.1
cytoolz                      0.11.2
dasbus                       1.4
dbus-python                  1.2.18
decorator                    4.4.2
distro                       1.5.0
dnaio                        1.2.0
dnspython                    2.3.0
docker-pycreds               0.4.0
dotmap                       1.3.30
file-magic                   0.4.0
flatbuffers                  23.5.26
fonttools                    4.44.0
frozendict                   1.2
gast                         0.5.4
gitdb                        4.0.11
GitPython                    3.1.40
google-auth                  2.23.4
google-auth-oauthlib         1.0.0
google-pasta                 0.2.0
gpg                          1.15.1
grpcio                       1.59.2
gssapi                       1.6.9
gurobipy                     11.0.2
h5py                         3.10.0
idna                         3.4
importlib-metadata           6.8.0
importlib-resources          6.1.1
iniparse                     0.4
iotop                        0.6
ipaclient                    4.11.0
ipalib                       4.11.0
ipaplatform                  4.11.0
ipapython                    4.11.0
isal                         1.5.3
Jinja2                       2.11.3
joblib                       1.3.2
jwcrypto                     0.8
keras                        2.15.0
kiwisolver                   1.4.5
langtable                    0.0.54
Levenshtein                  0.23.0
libclang                     16.0.6
libcomps                     0.1.18
llvmlite                     0.42.0
lxml                         4.6.5
Markdown                     3.5.1
MarkupSafe                   2.1.3
matplotlib                   3.8.1
ml-dtypes                    0.2.0
netaddr                      0.8.0
netifaces                    0.10.6
networkx                     3.2.1
nftables                     0.1
numba                        0.59.1
numpy                        1.26.4
nvidia-cublas-cu11           11.11.3.6
nvidia-cublas-cu12           12.2.5.6
nvidia-cuda-cupti-cu11       11.8.87
nvidia-cuda-cupti-cu12       12.2.142
nvidia-cuda-nvcc-cu11        11.8.89
nvidia-cuda-nvcc-cu12        12.2.140
nvidia-cuda-nvrtc-cu12       12.2.140
nvidia-cuda-runtime-cu11     11.8.89
nvidia-cuda-runtime-cu12     12.2.140
nvidia-cudnn-cu11            8.7.0.84
nvidia-cudnn-cu12            8.9.4.25
nvidia-cufft-cu11            10.9.0.58
nvidia-cufft-cu12            11.0.8.103
nvidia-curand-cu11           10.3.0.86
nvidia-curand-cu12           10.3.3.141
nvidia-cusolver-cu11         11.4.1.48
nvidia-cusolver-cu12         11.5.2.141
nvidia-cusparse-cu11         11.7.5.86
nvidia-cusparse-cu12         12.1.2.141
nvidia-nccl-cu11             2.16.5
nvidia-nccl-cu12             2.16.5
nvidia-nvjitlink-cu12        12.2.140
oauthlib                     3.2.2
opt-einsum                   3.3.0
packaging                    23.2
pandas                       2.1.2
perf                         0.1
pexpect                      4.8.0
pid                          2.2.3
Pillow                       10.1.0
pip                          24.0
ply                          3.11
productmd                    1.31
protobuf                     4.23.4
psutil                       5.9.6
ptyprocess                   0.6.0
pwquality                    1.4.4
py-cpuinfo                   8.0.0
pyasn1                       0.5.0
pyasn1-modules               0.3.0
pycairo                      1.20.1
pycosat                      0.6.3
pycparser                    2.20
pycups                       2.0.1
pycurl                       7.43.0.6
PyGObject                    3.40.1
pyinotify                    0.9.6
pykickstart                  3.32.11
pyOpenSSL                    21.0.0
pyparsing                    3.1.1
pyparted                     3.12.0
PySocks                      1.7.1
python-augeas                0.5.0
python-dateutil              2.8.2
python-ldap                  3.4.3
python-Levenshtein           0.23.0
python-linux-procfs          0.7.3
python-meh                   0.50
python-yubico                1.3.3
pytz                         2023.3.post1
pyudev                       0.22.0
pyusb                        1.0.2
PyYAML                       6.0.1
qrcode                       6.1
rapidfuzz                    3.5.2
requests                     2.31.0
requests-file                1.5.1
requests-ftp                 0.3.1
requests-oauthlib            1.3.1
rpm                          4.16.1.3
rsa                          4.9
ruamel.yaml                  0.16.6
ruamel.yaml.clib             0.2.7
scikit-learn                 1.3.2
scipy                        1.13.0
selinux                      3.6
sentry-sdk                   1.38.0
sepolicy                     3.6
setools                      4.4.4
setproctitle                 1.3.3
setroubleshoot               3.3.32
setuptools                   53.0.0
shap                         0.45.0
simpleline                   1.8.3
six                          1.16.0
slicer                       0.0.7
smmap                        5.0.1
sos                          4.7.2
SSSDConfig                   2.9.4
subscription-manager         1.29.40
systemd-python               234
tensorboard                  2.15.1
tensorboard-data-server      0.7.2
tensorflow                   2.15.0.post1
tensorflow-estimator         2.15.0
tensorflow-io-gcs-filesystem 0.34.0
tensorrt                     8.5.3.1
termcolor                    2.3.0
threadpoolctl                3.2.0
toolz                        0.11.2
tqdm                         4.65.0
typing_extensions            4.8.0
tzdata                       2023.3
urllib3                      2.0.7
wandb                        0.16.0
Werkzeug                     3.0.1
wheel                        0.41.3
wrapt                        1.14.1
xgboost                      2.0.1
xopen                        1.8.0
zipp                         3.17.0


