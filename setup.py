from setuptools import setup,find_packages

setup(
    name='WORD_CONTEXT',
    version='1.0',
    packages=find_packages(),
    url='',
    license='',
    author='Bedyouch Nidal',
    author_email='nidal.bedyouch.nb@gmail.com',
    description='Analyse words in their contexts to make theme classes.', install_requires=['nltk', 'langid',
                                                                                            'matplotlib', 'wordcloud',
                                                                                            'treetaggerwrapper',
                                                                                            'numpy', 'tagging']
)
