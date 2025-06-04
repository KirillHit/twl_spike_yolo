Configuration files
===================

Configuration files in the ``config`` folder define all parameters for training, validation, and testing.  
Each config is written in YAML format and is used to initialize model and dataset classes using two keys:

- ``class_path`` — specifies which class will be used (for example, for the model or dataset).
- ``init_args`` — sets the arguments that will be passed to the constructor of the selected class.

Parameters for the ``trainer`` section correspond to the arguments of the Lightning Trainer class.  
See the official documentation for all available options: `Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class>`_.

You can use the provided configs as templates or create your own.

Config file structure
--------------------

.. code-block:: yaml

    seed_everything: 42

    model:
      class_path: model.Yolo
      init_args:
        model: l
        num_classes: 2
        # ... other model parameters ...
        plotter:
          class_path: utils.Plotter
          init_args:
            threshold: 0.0
            show_video: true
            # ... other plotter parameters ...

    data:
      class_path: utils.PropheseeDataModule
      init_args:
        root: ./data/gen1
        batch_size: 8
        # ... other data parameters ...

    trainer:
      accelerator: gpu
      strategy: auto
      devices: -1
      # ... other trainer parameters ...

    # Optionally, you can add callbacks, logger, etc.

Available values for class_path
-------------------------------

For models:

- :class:`model.Yolo <model.nets.yolo.Yolo>`
- :class:`model.MultimodalYolo <model.nets.multimodal_yolo.MultimodalYolo>`

For datasets:

- :class:`utils.PropheseeDataModule <utils.dataset_prophesee.PropheseeDataModule>`
- :class:`utils.DSECDataModule <utils.dataset_dsec.DSECDataModule>`

For plotter:

- :class:`utils.Plotter <utils.plotter.Plotter>`

Parameter descriptions
----------------------

For a full list and description of available parameters for each class, see the API section:

- :class:`model.Detector <model.detector.Detector>`
- :class:`model.Yolo <model.nets.yolo.Yolo>`
- :class:`model.MultimodalYolo <model.nets.multimodal_yolo.MultimodalYolo>`
- :class:`utils.PropheseeDataModule <utils.dataset_prophesee.PropheseeDataModule>`
- :class:`utils.DSECDataModule <utils.dataset_dsec.DSECDataModule>`
- :class:`utils.Plotter <utils.plotter.Plotter>`

.. note::
   The classes :class:`model.Yolo <model.nets.yolo.Yolo>` and :class:`model.MultimodalYolo <model.nets.multimodal_yolo.MultimodalYolo>` both inherit from :class:`model.Detector <model.detector.Detector>`.  
   All initialization parameters of :class:`model.Detector <model.detector.Detector>` are also available for these models.

See the API section for constructor signatures and parameter documentation.
