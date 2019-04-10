from distutils.core import setup
setup(
  name = 'isacalc',         # How you named your package folder (MyLib)
  packages = ['isacalc'],   # Chose the same as "name"
  version = 'v1.0',      # Start with a small number and increase it with every change you make
  license='GNU GPLv3',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Standard International Atmosphere Calculator',   # Give a short description about your library
  author = 'Luke de Waal',                   # Type in your name
  author_email = 'lr.de.waal.01@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/LukeDeWaal/ISA_Calculator',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/LukeDeWaal/ISA_Calculator/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['ISA','Aerospace','Aeronautical','Atmosphere'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
