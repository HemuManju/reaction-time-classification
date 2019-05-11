Reaction_time_classification
==============================

In the teleoperated scenario, as the task difficulty increases, the performance of the operator decreases which leads to a decrease in the overall system efficiency. Thus, it is important to predict the change in task difficulty in order to increase system efficiency. However, the task difficulty cannot be predicted as task information is unknown in real-time. Alternatively, the task difficulty can be estimated studying the distribution of reaction time. In this study, the physiological features of the operator are used to classify the reaction time as fast, normal and slow corresponding different levels of task difficulty. The physiological features are extracted from the eye (though eye tracking) and brain (through Electroencephalogram) from the operator performing teleoperation using two drones. Among the calculated features glance ratio and mental workload resulted in maximum classification accuracy when task type information is included.

![Optional Text](/docs/tele_opreration_setup.png)


Project Organization
------------

    ├── src
    │   ├── config.yml
    │   ├── data
    │   │   ├── __init__.py
    │   │   └── create_dataset.py
    │   ├── features
    │   │   ├── __init__.py
    │   │   ├── features_selection.py
    │   │   └── utils.py
    │   ├── main.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── density_estimation.py
    │   │   ├── rt_classification.py
    │   │   ├── statistical_analysis.Rmd
    │   │   ├── t_sne_analysis.py
    │   │   ├── task_classification.py
    │   │   └── utils.py
    │   ├── utils.py
    │   └── visualization
            ├── __init__.py
            ├── utils.py
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
