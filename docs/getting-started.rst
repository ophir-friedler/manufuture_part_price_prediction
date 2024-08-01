This guideline assumes you've cloned the repository and are in the root directory.

Create the environment and activate
-----------------------------------

To create the environment, run the following command:

.. code:: bash

    make create_environment

This will create a virtual environment, which you should now activate:

.. code:: bash

    source activate manufuture_part_price_prediction

If you don't remember the activate command, you can run

.. code:: bash

    make activate_env


which will give you a copy&paste of the activation command.

Side note: if you're using an IDE like PyCharm, you can now set the
interpreter to the one in the virtual environment.
This will allow you to run the code from the IDE.

Install the requirements
------------------------

To install the requirements, run the following command:

.. code:: bash

    make requirements

This will install all the necessary packages.

Set up MySQL server
---------------------

Activate a local MySQL server. Make sure the manufuture database is created.
You can do this by running the following command:

.. code:: bash

    mysql -u root -p manufuture < manufuture_sql_db_dump.sql


Create a .env file
------------------

Create a .env file in the root directory of the project.
This file should contain the following (with example values):

.. code-block:: bash

    # .env
    MYSQL_HOST=localhost
    MYSQL_USER=root
    MYSQL_PASSWORD=mysql123
    LOCAL_USER=my_local_user_name

    PROJECT_NAME=manufuture_part_price_prediction
    PROJECT_DIR=/Users/${LOCAL_USER}/dev/${PROJECT_NAME}
    MY_CONDA_DIR=/Users/${LOCAL_USER}/opt/anaconda3/envs/manufuture_part_price_prediction
    MY_CONDA_ACTIVATE=/Users/${LOCAL_USER}/anaconda3/bin/activate

Prepare the RnD database
------------------------

To prepare the database, run the following command:

.. code:: bash

    make prepare_mysql

This will create the database and the tables.


Make sure all external data is in place
---------------------------------------

Specifically - werk data needs to be in the `data/external/werk_data` directory.
Manufuture CSVs are in the `data/external/mf_data` directory.

Process data
------------


.. code:: bash

     make tidy_data

This will read the data from manufuture database, and write it to parquets.
It will also read the werk data and write it to parquets.
Finally, it will write all necessary data to the rnd database for the model to use.


Train the model
---------------
To train the model, run the following command:

.. code:: bash

    make train_model

This will train the model on 80% of the data and save it to the `models` directory.


Evaluate the model
------------------
To evaluate the model, run the following command:

.. code:: bash

    make evaluate_model MODEL_NAME=your_model_name

This will evaluate the model on the remaining 20% of the data.

See model details
-----------------
To see the details of the model, run the following command:

.. code:: bash

    make model_details MODEL_NAME=your_model_name

This will show the details of the model, including the input features needed.