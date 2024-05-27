import sys
import sysconfig
import argparse

parser = argparse.ArgumentParser(description='Platlib path')
parser.add_argument('prefix', type=str, help='prefix path')
args = parser.parse_args()
pfx = args.prefix

try:
    # override scheme on debian/ubuntu py3.10, where 'posix_local' is set and malfunctioning.
    if sysconfig.get_default_scheme() == "posix_local":
        print(
            sysconfig.get_path(
                "platlib",
                vars={} if pfx == "" else {"base": pfx, "platbase": pfx},
                scheme="posix_prefix",
            )
        )
        sys.exit()
except AttributeError:
    # we're on Python <= 3.9, no scheme setting required and get_default_scheme does not exist.
    pass
print(
    sysconfig.get_path(
        "platlib", vars={} if pfx == "" else {"base": pfx, "platbase": pfx}
    )
)
