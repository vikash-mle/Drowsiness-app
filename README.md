# Drowsiness-app
Hi there, 
This repository contains files for building and deploying a computer vision model on GCP.
## api-service
This contains code for serving apis for various modules such as model serving module and api building service module. In order to build and run this container, run docker-shell.bat for windows or docker-shell.sh for linux in the CLI.
## frontend
it contains code to build web app interface. to build this container first run Docker.dev for development purpose and testing in local system. For deployment run Dockerfile.
## deployment
This container is for the sole purpose of connnecting with GCP, creating VMs and deploying and managing containers. We have used ansible for automation of all these tasks. Once you run this contaier, you thave to follow these steps to build, push and deploy containers on GCP:
### API's to enable in GCP
* Compute Engine API
* Service Usage API
* Cloud Resource Manager API
* Google Container Registry API

### Create a service account for deployment

- Go to [GCP Console](https://console.cloud.google.com/home/dashboard), and create a"Service accounts" called "deployment"
- Give it following permission:
    - Compute Admin
    - Compute OS Login
    - Container Registry Service Agent
    - Kubernetes Engine Admin
    - Service Account User
    - Storage Admin
  
### Building and deploying containers
- cd into deployment directory and run docker-shell.bat for windows or docker-shell.sh for linux. Once done , you will be connected to your GCP rewsources. Next is to build push conatiners to be deployed.
- Configure os login to be able to SSH into your VM
- Update your inventory.yml file for user and other compute instance specification
- Run - "ansible-playbook deploy-docker-images.yml -i inventory.yml" inside your deployment container to build and push images to GCR
- Then, run "ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present to create a VM instance 
- Run "ansible-playbook deploy-provision-instance.yml -i inventory.yml" to install and update various dependencies 
- Run "ansible-playbook deploy-setup-containers.yml -i inventory.yml" to setup containers in the VM
- Run "ansible-playbook deploy-setup-webserver.yml -i inventory.yml" to setup web server. After running this go to external_IP of your VM. What do you see?
## Thank you
