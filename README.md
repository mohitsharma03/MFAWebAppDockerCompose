This is a Multi-Factor-Authentication Web App based on flask which uses mysql db.
The App will create two docker containers: one for flask app and the other for mysql db.
'Docker Compose' is used to manage both the containers and then make them communicate with each other.

run below command to create and run the containers(it will work after you have installed docker engine & docker compose in your system):

docker compose up
