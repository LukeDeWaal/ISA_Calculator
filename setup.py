from distutils.core import setup


def readme():
  with open('README.md') as f:
    return f.read()


version = 'v1.3.2'

setup(
  name = 'isacalc',         # How you named your package folder (MyLib)
  packages = ['isacalc'],   # Chose the same as "name"
  package_data={'' : ['**/*.json']},
  version = version,      # Start with a small number and increase it with every change you make
  license='GNU GPLv3',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Standard International Atmosphere Calculator',   # Give a short description about your library
  long_description=readme(),
  long_description_content_type='text/plain',
  author = 'Luke de Waal',                   # Type in your name
  author_email = 'lr.de.waal.01@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/LukeDeWaal/ISA_Calculator',   # Provide either the link to your github or to your website
  download_url = f'https://github.com/LukeDeWaal/ISA_Calculator/archive/{version}.tar.gz',    # I explain this later on
  keywords = ['ISA','Aerospace','Aeronautical','Atmosphere'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy', 'pandas', 'tabulate'
      ],
  include_package_data=True,
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   # Again, pick a license
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
  ],
)
