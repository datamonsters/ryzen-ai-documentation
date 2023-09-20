# Installing IPU Driver

Uninstall default/old drivers:
- Check Device Manager → System devices → "AMD IPU Device". If device is there, uninstall it
- Right click "AMD IPU Device" and choose Uninstall device from the menu
  - If available, check the "delete the driver software for the device" box and click Uninstall

Install our custom AMD IPU Device:
- Go to Desktop\ipu_stack* and double click amd_install_kipudrv.bat.
- Check Device Manager → System devices → AMD IPU Device → properties → Driver.
