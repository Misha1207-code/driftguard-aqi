pipeline {
  agent any
  options { timestamps() }
  parameters {
    booleanParam(name: 'RUN_REPLAY', defaultValue: false,
                 description: 'Run the full policy-replay experiment (a few minutes)')
  }
  triggers { pollSCM('H/5 * * * *') }
  environment { IMAGE = "driftguard-aqi:${env.BUILD_NUMBER}" }
  stages {
    stage('Checkout') { steps { checkout scm } }
    stage('Install') {
      steps { sh 'python3 -m venv .venv && . .venv/bin/activate && pip install -q -r requirements.txt' }
    }
    stage('Test') {
      steps { sh '. .venv/bin/activate && pytest -q --junitxml=test-results.xml' }
      post { always { junit 'test-results.xml' } }
    }
    stage('Replay experiment') {
      when { expression { params.RUN_REPLAY } }
      steps {
        sh '. .venv/bin/activate && python driftguard.py --out results'
        archiveArtifacts artifacts: 'results/**', fingerprint: true
      }
    }
    stage('Build image') {
      steps { sh 'docker build -t ${IMAGE} -t driftguard-aqi:latest .' }
    }
    stage('Smoke test') {
      steps {
        sh '''
          docker rm -f dg-smoke || true
          docker run -d --name dg-smoke -p 8001:8000 ${IMAGE}
          for i in $(seq 1 40); do curl -sf localhost:8001/health && break; sleep 3; done
          curl -sf -X POST localhost:8001/simulate/step
          curl -sf localhost:8001/metrics | grep driftguard_retrain_score
        '''
      }
      post { always { sh 'docker rm -f dg-smoke || true' } }
    }
    stage('Deploy') {
      steps { sh 'docker compose up -d --build' }
    }
  }
  post {
    success { echo 'Deployed: app :8000, Prometheus :9090, Grafana :3000' }
    failure { echo 'Pipeline failed: check the stage logs' }
  }
}
