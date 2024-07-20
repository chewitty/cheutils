# cheutils

[cheutils](https://github.com/chewitty/cheutils) is a set of basic reusable utilities to and tools to facilitate quickly getting up and going on any project.

### Features

- propertiesutil: Using properties files to configure applications etc.
- baseutils: a common approach to project folder structure.
- stringutils: string processing utilities e.g., appending date pattern to file names

### Usage

```
import cheutils

# append date pattern to file name
cheutils.stringutils.datestamped('test.txt')  # returns something like test-2024-03-06.txt

```
