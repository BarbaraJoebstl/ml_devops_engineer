%md

# Data provenance

...is the complete origin, movement, and manipulation of data.

Origin can include not just where the data was retrieved from but also how the data set was generated, e.g. Census data retrieved from their API and that data was gathered by the American Community Survey in a particular year.

Movement may not always apply, but when it does it would encompass data that you or your team received from someone else. In other words, it's the origin plus any other jumps the data has made, e.g. data pulled from an API and moved to a company's S3 bucket for storage only to later be moved to HDFS for analysis.

Lastly is manipulation. Data may come to you completely raw, and then the manipulation will be any transforms or alterations you do on the data. Alternatively, data may come to you in a baked form and the various transformations will ideally be documented. A common example in NLP is documentation on how a data set is processed such as by changing the case or removing numbers or punctuation.

see also: https://en.wikipedia.org/wiki/Data_lineage

# DVC - Data Version Control
Data Version Control (DVC) is a complete solution for managing data, models, and the process of going from data to the model. All the while, it integrates nicely with tools that we already use (or intend to use) such as git and Continuous Integration/Continuous Deployment (CI/CD).

DVC's name is a bit of a misnomer in that it goes well beyond versioning data, and technically it does not even version data at all! Instead, it uses git for the actual versioning. DVC leverages a remote storage to hold the data but then tracks a record file using git (e.g. data.csv.dvc).

Beyond data versioning, DVC is a full experiment management system through its pipeline functionality. In DVC you can define a reusable pipeline (which is also version controlled). These pipelines can be used to build a reproducible model workflow and can be written so experiments can be logged and compared to help choose the best deployment model.

See also: https://dvc.org/doc/start

"""
mkdir /local/remote
dvc remote add -d localremote /local/remote
"""
DVC can be used entirely locally, and that's a great way to learn it! But the true power of DVC is unlocked when you set up remote storage for your project.

Remote storage enables you to use the same data/models regardless of what machine you are working on and allows you to easily share data and models with others. In other words, it makes sharing code/models as easy as it is to git clone a repository.

DVC conveniently provides a multitude of ways to retrieve remotely tracked data/models. This enables one to pull in data while working outside of a DVC project, or to easily pull data into the environment where a model may be deployed such as Heroku.