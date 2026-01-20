# Kaapi - Deployment

Kaapi uses a modern cloud-native deployment architecture built on AWS services with automated CI/CD pipelines.

## Deployment Architecture

### Overview

The deployment follows a containerized approach where:
- Application code is packaged into Docker images
- Images are stored in AWS ECR (Elastic Container Registry)
- ECS (Elastic Container Service) runs and manages the containers
- GitHub Actions automates the build and deployment process

### CI/CD Pipeline

The deployment pipeline is triggered automatically:
1. **Code Push**: Developer pushes code to GitHub
2. **Build**: GitHub Actions builds Docker image
3. **Push**: Image is pushed to ECR
4. **Deploy**: ECS pulls new image and updates running tasks

### Environments

Two deployment environments are configured:
- **Staging**: Deployed on every push to `main` branch for testing
- **Production**: Deployed only on version tags for stable releases

## Prerequisites

Before deploying, ensure the following AWS infrastructure exists:

### AWS Infrastructure

1. **ECS Clusters**: Separate clusters for staging and production environments
2. **ECR Repositories**: Container image repositories for each environment
3. **ECS Task Definitions**: Define container configurations, resource limits, and environment variables
4. **ECS Services**: Manage the desired number of running tasks
5. **IAM Role for GitHub**: Allows GitHub Actions to authenticate via OIDC (no long-lived credentials)
6. **RDS PostgreSQL**: Managed database service (recommended for production)
7. **ElastiCache Redis**: Managed Redis for caching (optional)
8. **Amazon MQ RabbitMQ**: Managed message broker for Celery (optional)

### GitHub Setup

Configure GitHub to allow automated deployments:

1. **Environment**: Create `AWS_ENV_VARS` environment for deployment protection
2. **Variable**: Set `AWS_RESOURCE_PREFIX` to identify your AWS resources

## AWS Resource Naming Convention

All AWS resources follow a consistent naming pattern using `AWS_RESOURCE_PREFIX` as the base identifier.

This naming convention ensures clear separation between environments and easy identification of resources.

## Deployment Workflows

### Staging Deployment

**Purpose**: Automatically deploy changes to a testing environment for validation before production.

**Trigger**: Push to `main` branch

```bash
git push origin main
```

**Workflow Steps** (`.github/workflows/cd-staging.yml`):
1. **Checkout**: Clone the repository code
2. **AWS Authentication**: Use OIDC to authenticate (no stored credentials)
3. **ECR Login**: Authenticate to container registry
4. **Build**: Create Docker image from `./backend` directory
5. **Push**: Upload image to staging ECR repository with `latest` tag
6. **Deploy**: Force ECS to pull and deploy the new image

The deployment typically completes in 5-10 minutes depending on image size and ECS configuration.

### Production Deployment

**Purpose**: Deploy stable, tested versions to the production environment.

**Trigger**: Create and push a version tag

```bash
# Create a version tag
git tag v1.0.0

# Push the tag to trigger deployment
git push origin v1.0.0
```

**Workflow Steps** (`.github/workflows/cd-production.yml`):
1. **Checkout**: Clone the repository at the tagged version
2. **AWS Authentication**: Use OIDC to authenticate
3. **ECR Login**: Authenticate to container registry
4. **Build**: Create Docker image from `./backend` directory
5. **Push**: Upload image to production ECR repository with `latest` tag
6. **Deploy**: Force ECS to pull and deploy the new image

**Best Practice**: Use semantic versioning (e.g., `v1.0.0`, `v1.2.3`) to clearly identify releases.

## GitHub Configuration

### Step 1: Create Environment

Environments in GitHub provide deployment protection and organization.

1. Go to repository **Settings → Environments**
2. Click **New environment**
3. Name it: `AWS_ENV_VARS`
4. Optionally, add protection rules (e.g., required reviewers)

### Step 2: Set Repository Variable

Variables store non-sensitive configuration that workflows need.

1. Go to **Settings → Secrets and variables → Actions → Variables tab**
2. Click **New repository variable**
3. Add:
   - **Name**: `AWS_RESOURCE_PREFIX`
   - **Value**: Your AWS resource prefix (e.g., `kaapi`)

### Step 3: AWS Authentication Setup

The workflows use **AWS OIDC authentication**, which is more secure than storing AWS access keys:
- No long-lived credentials stored in GitHub
- AWS IAM role assumes identity based on GitHub's OIDC token
- Permissions are scoped to specific actions

The IAM role ARN is configured in workflow files:

```yaml
role-to-assume: arn:aws:iam::{YOUR_AWS_ACCOUNT_ID}:role/github-action-role
aws-region: {YOUR_AWS_REGION}
```

**Note**: Replace `{YOUR_AWS_ACCOUNT_ID}` with your AWS account ID and `{YOUR_AWS_REGION}` with your chosen region (e.g., `ap-south-1`, `us-east-1`).

## Environment Variables in AWS ECS

Application configuration is managed through environment variables set in **ECS Task Definitions**. These are injected into containers at runtime.

### Configuring Environment Variables

Environment variables in ECS Task Definitions include database credentials, AWS credentials, API keys, and service endpoints. Refer to `.env.example` in the repository for a complete list of required and optional variables.

Key categories:
- **Authentication & Security**: JWT keys, admin credentials
- **Database**: PostgreSQL connection details
- **AWS Services**: S3 access credentials
- **Background Tasks**: RabbitMQ and Redis endpoints
- **Optional**: OpenAI API key, Sentry DSN

### Generate Secure Keys

Use Python to generate cryptographically secure keys:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Run this multiple times to generate different keys for `SECRET_KEY`, passwords, etc.

## Database Migrations

Database schema changes must be applied before deploying new application versions. This ensures the database structure matches what the code expects.

### Using ECS Run Task (Recommended for Production)

Run migrations as a one-time ECS task:

```bash
aws ecs run-task \
  --cluster {prefix}-cluster \
  --task-definition {migration-task-def} \
  --region {YOUR_AWS_REGION}
```

This runs the migration in the same environment as your application, ensuring consistency.

### Local Migration (Development/Testing)

For testing migrations locally:

```bash
cd backend
uv run alembic upgrade head
```

**Important**: Always test migrations in staging before applying to production.

## Monitoring & Observability

### AWS CloudWatch

**Logs**: View application logs from ECS tasks
```bash
aws logs tail /ecs/{cluster}/{service} --follow
```

**Metrics**: Monitor CPU, memory, request count, error rates

### ECS Console

- View running tasks and their health status
- Check deployment status and history
- Monitor service events and errors

### Health Checks

ECS performs health checks on the `/api/v1/utils/health/` endpoint. If this fails, tasks are replaced automatically.

## Rollback Procedures

If a deployment introduces issues, rollback to a previous stable version.

### List Previous Versions

```bash
aws ecs list-task-definitions --family-prefix {prefix}
```

### Rollback to Previous Version

```bash
aws ecs update-service \
  --cluster {prefix}-cluster \
  --service {prefix}-service \
  --task-definition {previous-task-def-arn} \
  --region {YOUR_AWS_REGION}
```

ECS will perform a rolling update back to the specified task definition.

**Tip**: Keep track of stable task definition ARNs for quick rollbacks.
