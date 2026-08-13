{{- define "rigorousrag.name" -}}
rigorousrag
{{- end -}}

{{- define "rigorousrag.fullname" -}}
{{- printf "%s" (include "rigorousrag.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rigorousrag.labels" -}}
app.kubernetes.io/name: {{ include "rigorousrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "rigorousrag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rigorousrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
